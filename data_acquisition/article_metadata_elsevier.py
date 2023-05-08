from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch


# Replace YOUR_API_KEY with your actual API key
api_key = "YOUR_API_KEY"
api_key = "1c5b12f12c61666c2c136780b427a25f"

## Initialize client
client = ElsClient(api_key)

scp_doc = AbsDoc(uri = 'doi/10.1016/S1525-1578(10)60571-5')
if scp_doc.read(client):
    print ("scp_doc.title: ", scp_doc.title)
    scp_doc.write()
else:
    print ("Read document failed.")

doi_doc = FullDoc(doi = '10.1111/csp2.12853')
if doi_doc.read(client):
    print ("doi_doc.title: ", doi_doc.title)
    doi_doc.write()
else:
    print ("Read document failed.")
