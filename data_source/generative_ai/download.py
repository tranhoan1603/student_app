import os 
import wget

file_links = [
    {
        'title' : 'A Survey of Large Language Models',
        'url'   : 'https://arxiv.org/pdf/2303.18223'
    }
]

def is_exist(file_link):
    return os.path.exists(f'./{file_link['title']}.pdf')

for file_link in file_links:
    if not is_exist(file_link):
        wget.download(file_link['url'], out=f'./{file_link['title']}.pdf')

