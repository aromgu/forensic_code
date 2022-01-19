import smtplib
from email.mime.text import MIMEText
from utils import *

def send_email():
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login('22201350@inha.edu', 'sibal123!')
    msg = MIMEText('{model} {save_root} is finished! Wake up and check'.format(model = args.model_name, save_root=args.save_root))
    msg['Subject'] = 'GPU training FINISHED :D'
    s.sendmail('22201350@inha.edu', '22201350@inha.edu', msg.as_string())
    s.quit()