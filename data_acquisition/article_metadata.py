# %% imports
from semanticscholar import SemanticScholar
from openalexapi import OpenAlex

# %% doi
# doi = 'doi:10.1111/csp2.12853'
# doi = 'doi:10.1007/s11356-022-24013-5'
doi = 'doi:10.3390/jof8121274'

# %% semantic Scholar example
scholar = SemanticScholar()
try:
    paper = scholar.get_paper(doi)
    print('************************')
    print('Semantic Scholar Example')
    print('************************')
    print('Title : ', paper.title)
    print('Open Access :', paper.isOpenAccess)
    print('First Author :', paper.authors[0].name, ' ', paper.authors[0].affiliations)
    print('abstract :', paper.abstract)

    for author in paper.authors:
        print(author.name, ' ', author.affiliations)
except Exception as e:
    print("Error ", e)

# %% semantic Open Alex example
openalex = OpenAlex()
openalex.email = "l.mwanzia@cgiar.org"
try:
    oapaper = openalex.get_single_work(doi)
    print('************************')
    print('Open Alex Example')
    print('************************')
    print ('Title : ',oapaper.title)
    print('Open Access :', oapaper.open_access)
    for author in oapaper.authorships:
        print(author.author, ' ', author.author_position, ' ', author.institutions)
except Exception as e:
    print("Error ", e)


