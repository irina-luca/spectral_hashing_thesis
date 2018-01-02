# -- Last downloaded category is: sep1_seq3_bldg400_outdoor/ -- #


# import requests
# from bs4 import BeautifulSoup

# # from BeautifulSoup import BeautifulSoup as bs
# import urlparse
# from urllib2 import urlopen
# from urllib import urlretrieve
# import os
# import sys
#
# def main(url, out_folder="D:/deckfinal/"):
#     soup = BeautifulSoup(urlopen(url).read())
#     for image in soup.findAll("a"):
#         parsed = url+image['href']
#         filename = image['href']
#         outpath = os.path.join(out_folder, filename)
#         try:
#             urlretrieve(parsed, outpath)
#         except:
#             print("skipping" + parsed)
#
# if __name__ == "__main__":
#     parent_url = "http://people.csail.mit.edu/brussell/research/LabelMe/Images/"
#     r = requests.get(parent_url)
#     html = r.text
#     soup = BeautifulSoup(html, "lxml")
#     a_hrefs = soup.findAll("a")
#
#     for a_href in a_hrefs:
#         a_href_text = a_href.contents[0]
#         if "/" in a_href_text:
#             print(a_href_text)
#             main("http://people.csail.mit.edu/brussell/research/LabelMe/Images/" + a_href_text,
#                  "./LabelMeImages")