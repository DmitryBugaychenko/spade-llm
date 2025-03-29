import certifi
from gigachat import GigaChat
import os

def main():

    giga = GigaChat(
        credentials=os.environ['GIGA_CRED'],
        ca_bundle_file="/Users/dmitrybugaychenko/Documents/RCA.pem.pem"
    )

    response = giga.get_token()

    print(response.access_token)

if __name__ == "__main__":
    main()