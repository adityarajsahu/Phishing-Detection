import pandas as pd
import numpy as np

DataFrame = pd.read_csv('Dataset/Phising_Training_Dataset.csv')
print(DataFrame.shape)
print(DataFrame.columns)

print("having_IP: ", len(DataFrame[DataFrame["having_IP"].isin([-1, 1])]))
print("URL_Length: ", len(DataFrame[DataFrame["URL_Length"].isin([-1, 0, 1])]))
print("Shortining_Service: ", len(DataFrame[DataFrame["Shortining_Service"].isin([-1, 1])]))
print("having_At_Symbol: ", len(DataFrame[DataFrame["having_At_Symbol"].isin([-1, 1])]))
print("double_slash_redirecting: ", len(DataFrame[DataFrame["double_slash_redirecting"].isin([-1, 1])]))
print("Prefix_Suffix: ", len(DataFrame[DataFrame["Prefix_Suffix"].isin([-1, 1])]))
print("having_Sub_Domain: ", len(DataFrame[DataFrame["having_Sub_Domain"].isin([-1, 0, 1])]))
print("SSLfinal_State: ", len(DataFrame[DataFrame["SSLfinal_State"].isin([-1, 0, 1])]))
print("Domain_registeration_length: ", len(DataFrame[DataFrame["Domain_registeration_length"].isin([-1, 1])]))
print("Favicon: ", len(DataFrame[DataFrame["Favicon"].isin([-1, 1])]))
print("port: ", len(DataFrame[DataFrame["port"].isin([-1, 1])]))
print("HTTPS_token: ", len(DataFrame[DataFrame["HTTPS_token"].isin([-1, 1])]))
print("Request_URL: ", len(DataFrame[DataFrame["Request_URL"].isin([-1, 1])]))
print("URL_of_Anchor: ", len(DataFrame[DataFrame["URL_of_Anchor"].isin([-1, 0, 1])]))
print("Links_in_tags: ", len(DataFrame[DataFrame["Links_in_tags"].isin([-1, 0, 1])]))
print("SFH: ", len(DataFrame[DataFrame["SFH"].isin([-1, 0, 1])]))
print("Submitting_to_email: ", len(DataFrame[DataFrame["Submitting_to_email"].isin([-1, 1])]))
print("Abnormal_URL: ", len(DataFrame[DataFrame["Abnormal_URL"].isin([-1, 1])]))
print("Redirect: ", len(DataFrame[DataFrame["Redirect"].isin([0, 1])]))
print("on_mouseover: ", len(DataFrame[DataFrame["on_mouseover"].isin([-1, 1])]))
print("RightClick: ", len(DataFrame[DataFrame["RightClick"].isin([-1, 1])]))
print("popUpWidnow: ", len(DataFrame[DataFrame["popUpWidnow"].isin([-1, 1])]))
print("Iframe: ", len(DataFrame[DataFrame["Iframe"].isin([-1, 1])]))
print("age_of_domain: ", len(DataFrame[DataFrame["age_of_domain"].isin([-1, 1])]))
print("DNSRecord: ", len(DataFrame[DataFrame["DNSRecord"].isin([-1, 1])]))
print("web_traffic: ", len(DataFrame[DataFrame["web_traffic"].isin([-1, 0, 1])]))
print("Page_Rank: ", len(DataFrame[DataFrame["Page_Rank"].isin([-1, 0, 1])]))
print("Google_Index: ", len(DataFrame[DataFrame["Google_Index"].isin([-1, 1])]))
print("Links_pointing_to_page: ", len(DataFrame[DataFrame["Links_pointing_to_page"].isin([-1, 0, 1])]))
print("Statistical_report: ", len(DataFrame[DataFrame["Statistical_report"].isin([-1, 1])]))


