from bs4 import BeautifulSoup
import requests

page = requests.get("https://en.wikipedia.org/wiki/Dutch_national_flag_problem")
soup = BeautifulSoup(page.content, 'html.parser')

allLinks = soup.find(id="bodyContent").find_all("a")

for link in allLinks:
    if(link['href'].find("/wiki/") == -1):
        continue
    try:
        print(link['title'])
    except:
        print("image")
