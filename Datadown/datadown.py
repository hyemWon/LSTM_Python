import requests

# 데이터 다운로드
class Download:
    url = ""
    file_name = ""

    def __init__(self, url, file_name):
        self.url = url
        self.file_name = file_name
        self.download()

    def download(self):
        r = requests.get(self.url, allow_redirects=True)
        open(self.file_name, 'wb').write(r.content)
