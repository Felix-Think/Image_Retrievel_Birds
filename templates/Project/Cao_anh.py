
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
import os
from bs4 import BeautifulSoup
#|%%--%%| <aWzKlKUqh9|Kt7TQs4Y32>

# Cài đặt Selenium WebDriver
driver = webdriver.Chrome()

# |%%--%%| <Kt7TQs4Y32|xk9FYyowZc>

# URL của trang Flickr
url = "https://www.flickr.com/search/?text=Muscicapidae"

#|%%--%%| <xk9FYyowZc|EeaM3hruWG>

# Điều hướng tới trang Flickr
driver.get(url)


# Tạo thư mục lưu ảnh
output_dir = "flickr_images1"
os.makedirs(output_dir, exist_ok=True)


# Cuộn trang để tải thêm ảnh
for _ in range(10):  # Cuộn 10 lần (có thể thay đổi)
    if len(driver.window_handles) > 0:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Chờ ảnh tải
    else:
        print("Cửa sổ trình duyệt đã bị đóng.")
        break



# |%%--%%| <EeaM3hruWG|65zljuZMxT>

# Gửi yêu cầu tới trang Flickr
response = requests.get(url)
if response.status_code != 200:
    print("Không thể truy cập trang web!")
else:
    print("Truy cập thành công!")

# |%%--%%| <65zljuZMxT|ATsysOBctk>

# Phân tích HTML của trang web
soup = BeautifulSoup(response.content, "html.parser")

# |%%--%%| <ATsysOBctk|77COFxKdmg>



# |%%--%%| <77COFxKdmg|94OK6oki5n>

# Tìm tất cả thẻ ảnh
image_tags = soup.find_all("img")
print(f"Tìm thấy {len(image_tags)} ảnh trên trang.")

#|%%--%%| <94OK6oki5n|jenzSe4wS6>

max_images = 100
download_count = 0

# |%%--%%| <jenzSe4wS6|rQywmeZHOR>

# Tải ảnh về
for idx, img_tag in enumerate(image_tags):
    try:
        # Lấy URL ảnh từ thuộc tính "src"
        img_url = img_tag.get("src")
        if not img_url:
            continue

        # Nếu URL thiếu scheme, thêm "https:" vào đầu
        if img_url.startswith("//"):
            img_url = "https:" + img_url

        # Tải ảnh
        img_data = requests.get(img_url).content
        img_name = f"{download_count}.jpg"
        with open(os.path.join(output_dir, img_name), "wb") as f:
            f.write(img_data)

        download_count += 1
        print(f"Đã tải {img_name}")
        if download_count >= max_images:
            break
    except Exception as e:
        print(f"Lỗi khi tải ảnh {download_count}: {e}")

print(f"Tải về thành công {download_count} ảnh.")

# Đóng trình duyệt
driver.quit()
#|%%--%%| <rQywmeZHOR|Cx8GVkB8Nd>

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
import os
from bs4 import BeautifulSoup

# Cài đặt Selenium WebDriver
driver = webdriver.Chrome()

# URL của trang Flickr
url = "https://www.flickr.com/search/?text=Ninox+scutulata"

# Điều hướng tới trang Flickr
driver.get(url)

# Tạo thư mục lưu ảnh
output_dir = "cumeo"
os.makedirs(output_dir, exist_ok=True)

# Cuộn trang để tải thêm ảnh
for _ in range(10):  # Cuộn 10 lần (có thể thay đổi)
    if len(driver.window_handles) > 0:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Chờ ảnh tải
    else:
        print("Cửa sổ trình duyệt đã bị đóng.")
        break

# Lấy HTML sau khi cuộn trang
page_source = driver.page_source

# Đóng trình duyệt
driver.quit()

# Phân tích HTML của trang web
soup = BeautifulSoup(page_source, "html.parser")

# Tìm tất cả thẻ ảnh
image_tags = soup.find_all("img")
print(f"Tìm thấy {len(image_tags)} ảnh trên trang.")

max_images = 150
download_count = 0
# Tải ảnh về
for idx, img_tag in enumerate(image_tags):
    try:
        # Lấy URL ảnh từ thuộc tính "src"
        img_url = img_tag.get("src")
        if not img_url:
            continue

        # Nếu URL thiếu scheme, thêm "https:" vào đầu
        if img_url.startswith("//"):
            img_url = "https:" + img_url

        # Tải ảnh
        img_data = requests.get(img_url).content
        img_name = f"{download_count}.jpg"
        with open(os.path.join(output_dir, img_name), "wb") as f:
            f.write(img_data)

        download_count += 1
        print(f"Đã tải {img_name}")
        if download_count >= max_images:
            break
    except Exception as e:
        print(f"Lỗi khi tải ảnh {download_count}: {e}")

print(f"Tải về thành công {download_count} ảnh.")

