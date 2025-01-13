from flask import Flask, render_template, redirect, url_for, flash, session, request
from flask_bootstrap import Bootstrap
from models import db, Student, Attendance, Admin
from werkzeug.utils import secure_filename
from forms import StudentForm
import cv2
import face_recognition
import os
import pandas as pd
import numpy as np
from flask import jsonify
import pickle
from PIL import Image
from flask import Response
import csv
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

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


@app.route('/')
def home():
    return render_template('home.html')

# Load known student images
known_images = []
known_ids = []

# Path to the folder containing student images
images_folder = app.config['UPLOAD_FOLDER']

for filename in os.listdir(images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_images.append(encoding)
        known_ids.append(filename.split(".")[0])  # Use filename as student ID

# Save the encoded data to a file
with open("EncodeFile.p", "wb") as file:
    pickle.dump([known_images, known_ids], file)

# Load encoded face data
with open("EncodeFile.p", "rb") as file:
    encodedFaceKnown, studentIDs = pickle.load(file)


# Initialize video capture with DirectShow backend
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)  # Set width to 1280 pixels
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)  # Set height to 720 pixels


def mark_attendance_db(student_id):
    # Check if the student exists
    student = Student.query.filter_by(admin_number=student_id).first()
    if student:
        try:
            # Create a new attendance record
            attendance = Attendance(student_id=student.id, status='Present')
            db.session.add(attendance)
            db.session.commit()
            flash(f'Attendance marked successfully for {student.name}!', 'success')
            return student
        except Exception as e:
            db.session.rollback()
            flash('Error marking attendance. Please try again.', 'error')
            return None
    else:
        flash('Student not registered. Please contact admin.', 'error')
        return None

def mark_attendance():
    while True:
        success, img = capture.read()

        if not success:
            print("Failed to grab frame from webcam")
            break

        # Resize and convert the image to RGB
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        faceCurrentFrame = face_recognition.face_locations(imgSmall)
        encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

        if faceCurrentFrame:
            for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
                # Compare the detected face with known faces
                matches = face_recognition.compare_faces(encodedFaceKnown, encodeFace)
                faceDistance = face_recognition.face_distance(encodedFaceKnown, encodeFace)
                matchIndex = np.argmin(faceDistance)

                if matches[matchIndex]:
                    # If a match is found, mark attendance
                    id = studentIDs[matchIndex]
                    print(f"Attendance marked for Student ID: {id}")

                    # Mark attendance using SQLAlchemy
                    mark_attendance_db(id)

                    # Draw a rectangle around the detected face
                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode(".jpg", img)
        if not ret:
            print("Failed to encode frame")
            break

        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg \r\n\r\n" + frame + b"\r\n")

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
        return redirect(url_for('login'))  # Changed from admin_login to login

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

            flash('Student added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))

        except Exception as e:
            db.session.rollback()
            flash(f'Error adding student: {str(e)}', 'error')
            print(f"Error: {str(e)}")  # For debugging
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

# Edit Student Route
@app.route('/admin/edit/<int:student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = Student.query.get_or_404(student_id)
    form = StudentForm(obj=student)

    if form.validate_on_submit():
        # Handle file upload
        file = form.image_path.data
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static/images/students/', filename)  # Save to upload folder
            file.save(file_path)
            student.image_path = f"images/students/{filename}"  # Update the relative path


        # Update other fields
        form.populate_obj(student)
        if form.password.data:
            student.set_password(form.password.data)  # Update password if provided
        db.session.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_student.html', form=form, student=student)

# Delete Student Route
@app.route('/admin/delete/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if 'admin_id' not in session:
        flash('Please log in to delete student records', 'error')
        return redirect(url_for('admin_login'))

    # Fetch the student
    student = Student.query.get_or_404(student_id)

    # Delete all attendance records for the student
    Attendance.query.filter_by(student_id=student.id).delete()

    # Delete the student
    db.session.delete(student)
    db.session.commit()

    flash('Student and related attendance records deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/search_student/<string:admin_number>')
def search_student(admin_number):
    student = Student.query.filter_by(admin_number=admin_number).first()
    if student:
        total_attendance = len([att for att in student.attendances if att.status == 'Present'])
        return jsonify({
            'success': True,
            'student': {
                'admin_number': student.admin_number,
                'name': student.name,
                'department': student.department,
                'total_attendance': total_attendance,
                'image_path': url_for('static', filename=student.image_path.replace('static/', '')) if student.image_path else None
            }
        })
    return jsonify({'success': False})

@app.route('/start_attendance')
def start_attendance():
    return render_template('start_attendance.html')

@app.route('/video_feed')
def video_feed():
    return Response(mark_attendance(), mimetype="multipart/x-mixed-replace; boundary=frame")

from io import StringIO
from datetime import datetime

@app.route('/mark_attendance_manual', methods=['POST'])
def mark_attendance_manual():
    data = request.get_json()
    student_id = data.get('student_id')
    
    student = Student.query.filter_by(admin_number=student_id).first()
    if student:
        attendance = Attendance(student_id=student.id, status='Present')
        db.session.add(attendance)
        db.session.commit()
        
        total_attendance = len([att for att in student.attendances if att.status == 'Present'])
        return jsonify({
            'success': True,
            'total_attendance': total_attendance
        })
    return jsonify({'success': False})

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create all database tables
        create_default_admin()  # Create default admin
    app.run(debug=True)