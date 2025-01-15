from flask import Flask, render_template, redirect, url_for, flash, session, request, Response, stream_with_context, jsonify
import json
from datetime import date
import csv
from io import StringIO
from datetime import datetime
import time
from sqlalchemy import func
from flask_bootstrap import Bootstrap
from models import db, Student, Attendance, Admin
from werkzeug.utils import secure_filename
from forms import StudentForm
import cv2
import face_recognition
import os
import numpy as np
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Face encoding variables
encodedFaceKnown = []
studentIDs = []

# Initialize camera capture
capture = None

def initialize_camera():
    try:
        print("Attempting to initialize camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow first
        
        if not cap.isOpened():
            print("DirectShow failed, trying default...")
            cap = cv2.VideoCapture(0)  # Try default
            
        if not cap.isOpened():
            print("Failed to open camera")
            return None
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Test camera
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera")
            cap.release()
            return None
            
        print("Camera initialized successfully")
        return cap
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None


# Initialize Flask application
app = Flask(__name__)

# Load configuration from config.py
app.config.from_pyfile('config.py')

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/images/students'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database and migrations
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Bootstrap
Bootstrap(app)

# Global variables for attendance tracking
already_marked_id_student = []
already_marked_id_admin = []
today_attendance = []  # Store today's attendance records




@app.route('/')
def home():
    return render_template('home.html')

def load_known_faces():
    """Load and encode faces from the images directory"""
    print("Starting to load known faces...")
    known_faces = []
    known_ids = []
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        print(f"ERROR: Upload folder {app.config['UPLOAD_FOLDER']} does not exist!")
        return [], []
    
    image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("ERROR: No image files found in upload folder!")
        return [], []
    
    print(f"Found {len(image_files)} image files")
    
    for filename in image_files:
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Processing {image_path}")
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                print(f"No face detected in {filename}")
                continue
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if face_encodings:
                known_faces.append(face_encodings[0])
                student_id = os.path.splitext(filename)[0]
                known_ids.append(student_id)
                print(f"Successfully encoded face for student ID: {student_id}")
            else:
                print(f"Could not encode face in {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Save encodings to file
    if known_faces:
        try:
            with open("EncodeFile.p", "wb") as file:
                pickle.dump([known_faces, known_ids], file)
            print(f"Successfully saved {len(known_faces)} face encodings")
        except Exception as e:
            print(f"Error saving encodings: {str(e)}")
    
    print(f"Successfully loaded {len(known_faces)} face encodings")
    return known_faces, known_ids

# Initialize face encodings
print("Initializing face encodings...")
try:
    if os.path.exists("EncodeFile.p"):
        with open("EncodeFile.p", "rb") as file:
            encodedFaceKnown, studentIDs = pickle.load(file)
            print(f"Loaded {len(encodedFaceKnown)} existing face encodings")
    else:
        print("No existing encodings found, generating new ones...")
        encodedFaceKnown, studentIDs = load_known_faces()
        if not encodedFaceKnown:
            print("WARNING: No face encodings were loaded!")
except Exception as e:
    print(f"Error loading face encodings: {str(e)}")
    encodedFaceKnown, studentIDs = [], []





def mark_attendance_db(student_id):
    print(f"Attempting to mark attendance for student ID: {student_id}")
    
    try:
        # Check if the student exists
        student = Student.query.filter_by(admin_number=student_id).first()
        if not student:
            print(f"Student with ID {student_id} not found")
            return None, 'Student not registered. Please contact admin.'

        # Check if attendance already marked for today
        today = datetime.now().date()
        existing_attendance = Attendance.query.filter(
            Attendance.student_id == student.id,
            func.date(Attendance.date) == today
        ).first()

        if existing_attendance:
            print(f"Attendance already marked for student {student.name} today")
            return student, "Attendance already marked for today!"
        
        try:
            # Create a new attendance record
            attendance = Attendance(
                student_id=student.id,
                status='Present',
                date=datetime.now()  # Explicitly set the date
            )
            db.session.add(attendance)
            db.session.commit()
            print(f"Successfully marked attendance for {student.name}")

            # Add to today's attendance list
            attendance_record = {
                'name': student.name,
                'department': student.department,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            global today_attendance
            today_attendance.append(attendance_record)
            
            # Update session with student info
            session['current_student'] = {
                'name': student.name,
                'department': student.department,
                'total_attendance': len([att for att in student.attendances if att.status == 'Present']),
                'admin_number': student.admin_number,
                'image_path': url_for('static', filename=student.image_path.replace('static/', '')) if student.image_path else None
            }
            
            # Clear the student after a short delay
            def clear_student():
                time.sleep(5)  # Keep student info for 5 seconds
                if 'current_student' in session:
                    del session['current_student']
            
            # Run the clear in background
            import threading
            threading.Thread(target=clear_student).start()
            
            return student, 'Attendance marked successfully!'
            
        except Exception as e:
            print(f"Database error while marking attendance: {str(e)}")
            db.session.rollback()
            return None, f'Error marking attendance: {str(e)}'
            
    except Exception as e:
        print(f"Unexpected error in mark_attendance_db: {str(e)}")
        if 'db' in locals():
            db.session.rollback()
        return None, 'An unexpected error occurred. Please try again.'

def mark_attendance():
    global capture, encodedFaceKnown, studentIDs
    
    print("Starting mark_attendance function...")
    
    if not encodedFaceKnown or not studentIDs:
        print("No known faces loaded. Please add student images first.")
        return
    
    if capture is None or not capture.isOpened():
        capture = initialize_camera()
        if capture is None:
            print("Error: Could not initialize camera")
            return
    
    while True:
        try:
            if capture is None or not capture.isOpened():
                print("Camera disconnected, attempting to reconnect...")
                capture = initialize_camera()
                if capture is None:
                    print("Failed to reconnect camera")
                    return
                
            success, frame = capture.read()
            if not success:
                print("Failed to grab frame, retrying...")
                time.sleep(1)
                continue
                
            # Process frame
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                print(f"Found {len(face_locations)} faces")
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Process each face found
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Scale back up face locations
                    top, right, bottom, left = [coord * 4 for coord in face_location]
                    
                    # Draw rectangle around face (default red for unknown)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    if len(encodedFaceKnown) > 0:
                        # Compare with known faces
                        matches = face_recognition.compare_faces(encodedFaceKnown, face_encoding, tolerance=0.6)
                        print(f"Face comparison results: {matches}")
                        
                        if True in matches:
                            match_index = matches.index(True)
                            student_id = studentIDs[match_index]
                            print(f"Matched with student ID: {student_id}")
                            
                            # Create app context for database operations
                            with app.app_context():
                                student, message = mark_attendance_db(student_id)
                                if student:
                                    # Change rectangle to green for recognized face
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    
                                    # Display student info and attendance status
                                    cv2.putText(frame, f"{student.name}", (left, top - 30),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                                    cv2.putText(frame, message, (left, top - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    print(f"Attendance status: {message}")
                                else:
                                    # Display error message
                                    cv2.putText(frame, message, (left, top - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                    print(f"Error marking attendance: {message}")
                        else:
                            print("No match found - Unknown face")
                            cv2.putText(frame, "Unknown", (left, top - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                            cv2.putText(frame, "Please register", (left, top - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Convert frame to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            if capture is not None:
                capture.release()
            capture = None
            time.sleep(1)
            continue








@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        role = request.form.get('role')

        username = request.form.get('username')
        password = request.form.get('password')

        if role == 'admin':
            admin = Admin.query.filter_by(username=username).first()
            if admin and admin.check_password(password):
                session['admin_id'] = admin.id
                flash('Admin login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin credentials', 'error')
                return redirect(url_for('login'))
        elif role == 'student':
            student = Student.query.filter_by(admin_number=username).first()
            if student and student.check_password(password):
                session['student_id'] = student.id
                flash('Student login successful!', 'success')
                return redirect(url_for('student_dashboard', admin_number=student.admin_number))
            else:
                flash('Invalid student credentials', 'error')
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/attendance')
def attendance():
    # Fetch all attendance records with student details
    records = db.session.query(Attendance, Student).join(Student).order_by(Attendance.date.desc()).all()
    return render_template('attendance.html', records=records)




@app.route('/student/<string:admin_number>')
def student_dashboard(admin_number):
    # Fetch the student by admin_number
    student = Student.query.filter_by(admin_number=admin_number).first_or_404()
    
    # Fetch attendance records for the student
    attendances = student.attendances
    
    # Calculate total attendance and percentage
    total_classes = len(attendances)
    present_count = len([att for att in attendances if att.status == 'Present'])
    attendance_percentage = (present_count / total_classes * 100) if total_classes > 0 else 0
    
    return render_template('student_dashboard.html', 
                         student=student, 
                         attendances=attendances, 
                         total_attendance=present_count,
                         total_classes=total_classes,
                         attendance_percentage=attendance_percentage)





@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if 'admin_id' not in session:
        flash('Please log in to access the admin dashboard', 'error')
        return redirect(url_for('login')) 

    form = StudentForm()
    if form.validate_on_submit():
        try:
            # Handle file upload
            file = form.image_path.data
            image_path = None
            if file:
                filename = secure_filename(f"{form.admin_number.data}{os.path.splitext(file.filename)[1]}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_path = f"images/students/{filename}"  # Store relative path

            # Create student
            student = Student(
                admin_number=form.admin_number.data,
                name=form.name.data,
                dob=form.dob.data,
                department=form.department.data,
                year=int(form.year.data),
                email=form.email.data,
                address=form.address.data,
                notice=form.notice.data,
                image_path=image_path
            )
            
            # Set password
            student.set_password(form.password.data)

            # Save to database
            db.session.add(student)
            db.session.commit()

            # Create initial attendance record
            attendance = Attendance(
                student_id=student.id,
                status=form.attendance.data
            )
            db.session.add(attendance)
            db.session.commit()

            flash('Student added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error adding student: {str(e)}', 'error')
            print(f"Error: {str(e)}")  # For debugging
            return redirect(url_for('admin_dashboard'))

    # Fetch all students for display
    students = Student.query.all()
    return render_template('admin_dashboard.html', form=form, students=students)


@app.route('/admin/add_student', methods=['GET', 'POST'])
def add_student():
    if 'admin_id' not in session:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login'))

    form = StudentForm()
    if form.validate_on_submit():
        try:
            # Handle file upload
            file = form.image_path.data
            image_path = None
            if file:
                filename = secure_filename(f"{form.admin_number.data}{os.path.splitext(file.filename)[1]}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_path = f"images/students/{filename}"

                # Verify the uploaded image has a detectable face
                try:
                    image = face_recognition.load_image_file(file_path)
                    face_locations = face_recognition.face_locations(image)
                    if not face_locations:
                        os.remove(file_path)  # Remove the invalid image
                        flash('No face detected in the uploaded image. Please try again.', 'error')
                        return redirect(url_for('add_student'))
                except Exception as e:
                    os.remove(file_path)
                    flash('Error processing the image. Please try a different image.', 'error')
                    return redirect(url_for('add_student'))

            # Create student without total_attendance
            student = Student(
                admin_number=form.admin_number.data,
                name=form.name.data,
                dob=form.dob.data,
                department=form.department.data,
                year=int(form.year.data),
                email=form.email.data,
                address=form.address.data,
                notice=form.notice.data,
                image_path=image_path
            )
            
            student.set_password(form.password.data)
            db.session.add(student)
            db.session.commit()

            # Create initial attendance record
            attendance = Attendance(
                student_id=student.id,
                status=form.attendance.data
            )
            db.session.add(attendance)
            db.session.commit()

            # Refresh face encodings after adding new student
            known_images, known_ids = load_face_encodings()
            global encodedFaceKnown, studentIDs
            encodedFaceKnown = known_images
            studentIDs = known_ids

            flash('Student added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error adding student: {str(e)}', 'error')
            print(f"Error: {str(e)}")
            return redirect(url_for('add_student'))

    return render_template('add_student.html', form=form)



def create_default_admin():
    # Check if the default admin already exists
    admin = Admin.query.filter_by(username='admin').first()
    if not admin:
        # Create the default admin
        admin = Admin(username='admin')
        admin.set_password('admin123')  # Hash the password
        db.session.add(admin)
        db.session.commit()
        print("Default admin created successfully!")
    else:
        print("Default admin already exists.")

@app.route('/admin/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    if 'admin_id' not in session:
        flash('Please log in to edit student records', 'error')
        return redirect(url_for('login'))

    student = Student.query.get_or_404(student_id)
    form = StudentForm(obj=student)

    if form.validate_on_submit():
        try:
            # Handle file upload
            file = form.image_path.data
            if file and file.filename:
                # Delete old image if it exists
                if student.image_path:
                    old_image_path = os.path.join('static', student.image_path)
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                
                # Save new image
                filename = secure_filename(f"{form.admin_number.data}{os.path.splitext(file.filename)[1]}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                student.image_path = f"images/students/{filename}"

            # Update other fields
            student.admin_number = form.admin_number.data
            student.name = form.name.data
            student.dob = form.dob.data
            student.department = form.department.data
            student.year = form.year.data
            student.email = form.email.data
            student.address = form.address.data
            student.notice = form.notice.data

            if form.password.data:
                student.set_password(form.password.data)

            db.session.commit()

            # Reload face encodings after updating student image
            if file and file.filename:
                global encodedFaceKnown, studentIDs
                known_images, known_ids = load_face_encodings()
                encodedFaceKnown = known_images
                studentIDs = known_ids

            flash('Student updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error updating student: {str(e)}', 'error')
            print(f"Error updating student: {str(e)}")
            return redirect(url_for('edit_student', student_id=student_id))

    return render_template('edit_student.html', form=form, student=student)

@app.route('/admin/delete/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if 'admin_id' not in session:
        flash('Please log in to delete student records', 'error')
        return redirect(url_for('login'))

    try:
        # Fetch the student
        student = Student.query.get_or_404(student_id)

        # Delete student's image file if it exists
        if student.image_path:
            image_file_path = os.path.join('static', student.image_path)
            if os.path.exists(image_file_path):
                os.remove(image_file_path)

        # Delete all attendance records for the student
        Attendance.query.filter_by(student_id=student.id).delete()

        # Delete the student from database
        db.session.delete(student)
        db.session.commit()

        # Reload face encodings after deleting student
        global encodedFaceKnown, studentIDs
        known_images, known_ids = load_face_encodings()
        encodedFaceKnown = known_images
        studentIDs = known_ids

        flash('Student and related records deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting student: {str(e)}', 'error')
        print(f"Error deleting student: {str(e)}")
    
    return redirect(url_for('admin_dashboard'))


@app.route('/start_attendance', methods=['GET', 'POST'])
def start_attendance():
    global capture
    if request.method == 'POST':
        if capture is None:
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                flash('Failed to initialize camera', 'error')
                return redirect(url_for('home'))
            # Set camera properties
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return redirect(url_for('attendance_view'))
    return render_template('start_attendance.html')


@app.route('/video_feed')
def video_feed():
    """Route for the video feed"""
    return Response(mark_attendance(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

from io import StringIO
from datetime import datetime


@app.route('/student_updates')
def student_updates():
    def generate():
        while True:
            # Check if we have a student in the current frame
            if 'current_student' in session:
                student = session['current_student']
                data = {
                    'success': True,
                    'student': student
                }
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)  # Check every 500ms
    
    return Response(stream_with_context(generate()),
                   mimetype='text/event-stream')

@app.route('/get_current_student')
def get_current_student():
    if 'current_student' in session:
        return jsonify({
            'success': True,
            'student': session['current_student']
        })
    return jsonify({
        'success': False,
        'student': None
    })

@app.route('/download_attendance')
def download_attendance():
    # Create a string buffer to write the CSV data
    si = StringIO()
    cw = csv.writer(si)
    
    # Write headers
    cw.writerow(['Student ID', 'Name', 'Department', 'Timestamp'])
    
    # Get attendance records
    records = db.session.query(Attendance, Student).join(Student).order_by(Attendance.date.desc()).all()
    
    # Write data rows
    for attendance, student in records:
        cw.writerow([
            student.admin_number,
            student.name,
            student.department,
            attendance.date.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    output = si.getvalue()
    si.close()
    
    # Create the response
    filename = f"attendance_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={filename}',
            'Content-type': 'text/csv'
        }
    )

@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup resources on application shutdown"""
    global capture
    if capture is not None:
        capture.release()
        capture = None

@app.route('/today_attendance')
def today_attendance():
    """Route to display today's attendance records"""
    # Get today's date
    today = datetime.now().date()
    
    # Query database for today's attendance records
    attendance_records = db.session.query(
        Attendance, Student
    ).join(Student).filter(
        func.date(Attendance.date) == today
    ).all()
    
    # Format the records
    attendance_list = [{
        'student_name': student.name,
        'department': student.department,
        'timestamp': attendance.date.strftime('%Y-%m-%d %H:%M:%S')
    } for attendance, student in attendance_records]
    
    return render_template('today_attendance.html', attendance_records=attendance_list)



@app.route('/download_today_attendance')
def download_today_attendance():
    """Route to download today's attendance as CSV"""
    # Create a string buffer to write the CSV data
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['Student Name', 'Department', 'Time', 'Status'])
    
    # Get today's date
    today = datetime.now().date()
    
    # Query database for today's attendance records
    attendance_records = db.session.query(
        Attendance, Student
    ).join(Student).filter(
        func.date(Attendance.date) == today
    ).all()
    
    # Write records to CSV
    for attendance, student in attendance_records:
        cw.writerow([
            student.name,
            student.department,
            attendance.date.strftime('%Y-%m-%d %H:%M:%S'),
            'Present'
        ])
    
    output = si.getvalue()
    si.close()
    
    filename = f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
    
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={filename}',
            'Content-type': 'text/csv'
        }
    )


@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    """Route to clear today's attendance records"""
    try:
        global today_attendance
        today = date.today()
        
        # Clear both database records and in-memory list
        Attendance.query.filter(
            func.date(Attendance.date) == today
        ).delete(synchronize_session=False)
        db.session.commit()
        
        today_attendance = []
        
        return jsonify({'success': True, 'message': 'Attendance cleared successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create all database tables
        create_default_admin()  # Create default admin
    app.run(debug=True)