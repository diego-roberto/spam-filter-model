import os
import email
from email.policy import default
import pandas as pd

# executar no mesmo diretório dos arquivos baixados e extraidos #

def read_emails_from_dir(directory, label):
    data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'rb') as file:
                    msg = email.message_from_binary_file(file, policy=default)
                    subject = msg['subject'] or ''
                    body = ''
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                body += part.get_payload(decode=True).decode(errors='ignore')
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')
                    data.append({
                        'subject': subject.strip(),
                        'body': body.strip(),
                        'label': label
                    })
            except Exception as e:
                print(f"Erro ao processar {filepath}: {e}")
    return data

all_emails = []
for label in ['easy_ham', 'hard_ham', 'spam']:
    spam_label = 'spam' if label == 'spam' else 'not spam'
    all_emails.extend(read_emails_from_dir(label, spam_label))

df = pd.DataFrame(all_emails)
df.to_csv('emails_preparados.csv', index=False)
print(df.head())
print(df['label'].value_counts())
