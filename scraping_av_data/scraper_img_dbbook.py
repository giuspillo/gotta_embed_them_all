import requests 
from bs4 import BeautifulSoup
import pandas as pd
import urllib.request
import time
import pickle as pkl
import os

def convert_url(dbpedia_url):
  dbpart = dbpedia_url.split('/')[-1]
  return 'https://en.wikipedia.org/wiki/'+dbpart

dburls = pd.read_csv("dbbook_items.tsv", sep='\t', header=None)

for _, row in dburls.iterrows():
  
  item_id = row[0]
  dburl = row[1]
  
  try:

    wikiurl = convert_url(dburl)

    filename = 'Images/' + str(item_id)+'.jpg'


    if not os.path.exists(filename):
      r = requests.get(wikiurl) 
        
      soup = BeautifulSoup(r.content, 'html5lib')

      img = soup.find('span', attrs = {'class': 'mw-default-size'}).find('img')['src']
      img_url = 'https:'+ img.strip().split(' ')[0]
      urllib.request.urlretrieve(img_url, filename)
      
      with open('id_db_img_wiki_dbbook.tsv', 'a') as tsvfile:
        line = str(item_id) + '\t' + dburl + '\t' + img_url + '\t' + wikiurl + '\n'
        tsvfile.write(line)

      print('done:' + wikiurl)
      time.sleep(1)

    else:

      print('existing: ' + wikiurl)
     
  except Exception as e:

    try:

        wikiurl = convert_url(dburl)

        filename = 'Images/' + str(item_id)+'.jpg'


        if not os.path.exists(filename):
          r = requests.get(wikiurl) 
            
          soup = BeautifulSoup(r.content, 'html5lib')

          img = soup.find('td', attrs = {'class': 'infobox-image'}).find('img')['src']
          img_url = 'https:'+ img.strip().split(' ')[0]
          urllib.request.urlretrieve(img_url, filename)
          
          with open('id_db_img_wiki_dbbook.tsv', 'a') as tsvfile:
            line = str(item_id) + '\t' + dburl + '\t' + img_url + '\t'+ wikiurl + '\n'
            tsvfile.write(line)


          print('done:' + wikiurl)
          time.sleep(1)

        else:

          print('existing: ' + wikiurl)
      
    except Exception as e:
      
      
      try:

        wikiurl = convert_url(dburl)+'_(novel)'

        filename = 'Images/' + str(item_id)+'.jpg'


        if not os.path.exists(filename):
          r = requests.get(wikiurl) 
          
          soup = BeautifulSoup(r.content, 'html5lib')
          img = soup.find('td', attrs = {'class': 'infobox-image'}).find('img')['src']
          img_url = 'https:'+ img.strip().split(' ')[0]
          urllib.request.urlretrieve(img_url, filename)
        
          with open('id_db_img_wiki_dbbook.tsv', 'a') as tsvfile:
            line = str(item_id) + '\t' + dburl + '\t' + img_url + '\t' + wikiurl + '\n'
            tsvfile.write(line)

          print('done:' + wikiurl)
          time.sleep(1)

        else:

          print('existing: ' + wikiurl)
      
        
      except Exception as e:

        try:

          wikiurl = convert_url(dburl)+'_(book)'

          filename = 'Images/' + str(item_id)+'.jpg'


          if not os.path.exists(filename):
            r = requests.get(wikiurl) 
          
            soup = BeautifulSoup(r.content, 'html5lib')
            img = soup.find('td', attrs = {'class': 'infobox-image'}).find('img')['src']
            img_url = 'https:'+ img.strip().split(' ')[0]
            urllib.request.urlretrieve(img_url, filename)
        
            with open('id_db_img_wiki_dbbook.tsv', 'a') as tsvfile:
              line = str(item_id) + '\t' + dburl + '\t' + img_url + '\t' + wikiurl + '\n'
              tsvfile.write(line)


            print('done:' + wikiurl)
            time.sleep(1)

          else:

            print('existing: ' + wikiurl)
      
        
        except Exception as e:
            
            with open('log.txt', 'a') as fout:
              line = 'ERROR with book ' + dburl + ' : ' + str(e) + '\n'
              fout.write(line)
              print(line)

            with open('NoCopertina.tsv', 'a') as tsvfile:
              line = str(item_id) + '\t' + dburl + '\n'
              tsvfile.write(line)

              time.sleep(1)

