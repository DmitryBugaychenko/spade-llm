import certifi
from gigachat import GigaChat
import os

def main():

    giga = GigaChat(
        auth_url="url для внутренней аутентификации",
        credentials=os.environ['GIGA_CRED'],
        verify_ssl_certs=False
    )

    response = giga.get_token()

    print(response.access_token)

if __name__ == "__main__":
    main()