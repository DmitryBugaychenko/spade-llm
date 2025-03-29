from gigachat import GigaChat
import os
from openai import OpenAI


def main():

    # giga = GigaChat(
    #     credentials=os.environ['GIGA_CRED'],
    #     verify_ssl_certs=False,
    # )
    #
    # response = giga.get_token()
    #
    # token = response.access_token
    token = "eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.I7kytvIlsjfT2uAjLB2nHpFiMfiZKp6F9HarzoLOJFxR1RdwVIlXrghi5bezfEH9LKzHBUQ_WoIy_15PMTc2tLImutOqYQRFfpyxRZz-DHPaBZI3nrw0k-D9EXXinAfpB3_0S1HA5Jbj2sRNYeCVRRH7GYd_tE-w_Feiwj2WrylImRyP0vJELvL0OZmcZKCKoHyCvlr0JTVDdZ-TlosjFEXXO47fHkAE8nLB5mmAZlG82TOt1hsdpHZ_lCUR24Fp0HsqtbtJKePQbOL1gmuEclCZS5EPGjaydZe_CMSS9M0oKhCFb5hsPLx3YgLVGrITkuD3LMCRJWbPFybxbVC48g.iZehFbxzDUYqumkidtK0hw.IKpTcY1e_jtcF3FVU6b9TU9Nmsta26ByQK6YMXrEufoXsQFpCJaNMTjZ61cUgdhjLyxDta5GaNbyNLITusdClzLQeDiDJBQrLP3uHDMVa_LGVWnUYwZz4u4Hxg-csOX_Nti5Wlcc5h_1Gt7f0LIqBh20fkQlQknekJLOrKjbGEJtxwYSQEZsH5NvoBHQmRLQljzfX4VihSyASPUnTFyatH9E8EjJUXIGKX5gNQGN0vl4tk1U8Icfqwc50R9dHrX9QbtMCN2LmDn9GQbws9j62wxybCOBwTFq-C_r0soGEUAq6RCdbueuppAxbbyJxANocxwyDTmdfjanyoeq4HyOByD6ywG72M0NKxYETR8ppOi5yJ_uNhfupWuBiTL5nyYwdvJt_e-JbOINfSj1gfQjRbzgOzhUHmN5jUCt0Nx8rgKEXf4jpK-QZzbUAR5MUz3QjphkCXls8XUX_3W6lRES1oosQ2YJfUV2t-5vmXjf7VSpHGYibQno8NhoZCbesOnv_unahpoWR-0obAmwN6JJI3KOOyEE461-f5ep0nFx_PtQq1hJxmWbxL0ZqXhPkY-BiMiTWX6VJRgpBdmFkolGxjkB3xcAe3dfFIiZVPOTrZGzd766gYNax3yMcoCw4XelD5PHEY0hxWFcoEJlS3ldaEXYc_nzPFClZcEIBZv8A4ZZRQGERu0MApZowuzXHVT152D0Vz7RbOTTpLOs8tvd3YSQyNPNaAdFcls8ClbGxsU.8hJdP5CjF3OFt-hei1cm7qv7lgnmStJ_3R58zVIXkDY"


    client = OpenAI(api_key= f"{token}",
                    base_url= "https://gigachat.devices.sberbank.ru/api/v1"

                    )
    completion = client.chat.completions.create(
        model="GigaChat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Hello World!"
            }
        ]
    )
    print(completion.choices[0].message)



if __name__ == "__main__":
    main()