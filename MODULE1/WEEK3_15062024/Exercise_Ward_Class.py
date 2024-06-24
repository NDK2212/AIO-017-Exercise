def compare(a_person):
    if isinstance(a_person, Student):
        return a_person.student_yob
    elif isinstance(a_person, Teacher):
        return a_person.teacher_yob
    elif isinstance(a_person, Doctor):
        return a_person.doctor_yob


class Ward:
    def __init__(self, name):
        self.ward_name = name
        self.list_of_people = []

    def add_person(self, a_person):
        self.list_of_people.append(a_person)

    def describe(self):
        for person in self.list_of_people:
            person.describe()

    def count_doctor(self):
        cnt_doc = 0
        for person in self.list_of_people:
            if isinstance(person, Doctor):
                cnt_doc += 1
        return cnt_doc

    def sort_age(self):
        self.list_of_people.sort(key=compare, reverse=True)

    def compute_average(self):
        sum_teacher_yob = 0
        sum_teacher = 0
        for person in self.list_of_people:
            if isinstance(person, Teacher):
                sum_teacher_yob += person.teacher_yob
                sum_teacher += 1
        return sum_teacher_yob / sum_teacher


class Student:
    def __init__(self, name, yob, grade):
        self.student_name = name
        self.student_yob = yob
        self.student_grade = grade

    def describe(self):
        print(
            f"Student - Name: {self.student_name}, YoB= {self.student_yob}, grade={self.student_grade}")


class Teacher:
    def __init__(self, name, yob, subject):
        self.teacher_name = name
        self.teacher_yob = yob
        self.teacher_subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self.teacher_name}, YoB= {self.teacher_yob}, grade={self.teacher_subject}")


class Doctor:
    def __init__(self, name, yob, specialist):
        self.doctor_name = name
        self.doctor_yob = yob
        self.doctor_specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self.doctor_name}, YoB= {self.doctor_yob}, grade={self.doctor_specialist}")
