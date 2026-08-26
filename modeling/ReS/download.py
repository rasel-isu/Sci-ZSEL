import requests
import os

# Replace the link with your shared OneDrive link
shared_link = "https://my.microsoftpersonalcontent.com/personal/86d1eb9ae5f6c984/_layouts/15/download.aspx?UniqueId=e5f6c984-eb9a-20d1-8086-7f0000000000&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiI5NGVlZTY2OC0wZWQ0LTRhYjgtYTEyZS1mYWI4N2ViMzI4ZmYiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbXkubWljcm9zb2Z0cGVyc29uYWxjb250ZW50LmNvbUA5MTg4MDQwZC02YzY3LTRjNWItYjExMi0zNmEzMDRiNjZkYWQiLCJleHAiOiIxNzQzMzcxODg3In0.fEpG26G2nBfsmr3pYqRqqmtCrGmBD4jArjyWrDAycyqyoNWcw3eotGiVOi3HwKd_uuuS2U0Es5Z-puUwQJv7JSks05i550Ucj75Sa7nNdz6yJaNy4Sn_wXfXNx0nWZI01u_0tW-02lo9RkU1Ei8DLAUB_6j7Veew4TAFXB7yZio87tWBjo_kwAO15Q9DroW2XBB3eyYVAQB_hQvnXhQMuajdELukP840CFpCAk1Xv-7M3EOpS1RySmvcfiWfIkK_vZKTvXk62PtzI_aKjyVpwe_NjrXkyOXGBkIl85xMGDGYQdfHNeTXQYfFFRrydLybs_CZCdYN0AmomDElCpTdE8mT05ygRFsmu4oGvxrcUILE9B4FD0iKlRQLLC-MhaM69FVbnR_xGnX7t4U8vURUfmfz85lEp_8Q_Lq_5ogq27xvvKVMywKSQuzbXCgxfXS272dSI9L0HvGurJZ0Dnki4iu7nHzxHkFFoGlW6c7PIt1IKjgoNM3oFEvQY0awMGV9_K-xO8_smU0KmtPlDP8iAw.zTTh6Y3I6bzd5N9Qtv30wwcXwsmDa2CkHQRxcOYJLT0&ApiVersion=2.0"

# The URL to extract file data (the shared link itself, shortened)
# You may need to extract the file IDs from the shared link, which could require parsing or using an API call to obtain file data.
# For simplicity, we assume you already have direct file download URLs.

# List of file download links (replace with actual links)
file_urls = [
    shared_link
]

# Folder to save downloaded files
download_folder = 'downloads'
os.makedirs(download_folder, exist_ok=True)

# Download each file
for url in file_urls:
    file_name = url.split('/')[-1]  # Extract file name from URL
    file_path = os.path.join(download_folder, file_name)
    
    # Download the file using requests
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name}")
    else:
        print(f"{response.status_code} Failed to download {file_name}")
