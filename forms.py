from flask_wtf import FlaskForm
from wtforms import StringField, DateField, SelectField, TextAreaField, FileField, IntegerField
from wtforms.validators import DataRequired, Email, Optional, NumberRange
from wtforms import StringField, EmailField , SubmitField
from wtforms.validators import DataRequired, Email
from wtforms import PasswordField, SubmitField



class StudentForm(FlaskForm):
    admin_number = StringField('Admin Number', validators=[DataRequired()])
    name = StringField('Name', validators=[DataRequired()])
    dob = DateField('Date of Birth', validators=[DataRequired()])
    department = StringField('Department', validators=[DataRequired()])
    year = SelectField('Year', choices=[('1', 'Year 1'), ('2', 'Year 2'), ('3', 'Year 3'), ('4', 'Year 4')], validators=[DataRequired()])  # Changed from level to year
    email = StringField('Email', validators=[DataRequired()])
    address = StringField('Address', validators=[DataRequired()])
    notice = TextAreaField('Notice')
    total_attendance = IntegerField('Total Attendance', validators=[Optional(), NumberRange(min=0)], default=0)
    image_path = FileField('Upload Image')
    password = PasswordField('Password', validators=[DataRequired()])
    attendance = SelectField('Initial Attendance Status', choices=[('Present', 'Present'), ('Absent', 'Absent')], validators=[DataRequired()])
    submit = SubmitField('Add Student')
