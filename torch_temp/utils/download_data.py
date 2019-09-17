
import errno
import os

from six.moves import urllib
import zipfile


def download(self):
    # download files
    try:
        os.makedirs(os.path.join(self.root, self.raw_folder))
        os.makedirs(os.path.join(self.root, self.processed_folder))
    except OSError as exception:
        if exception.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in self.urls:
        print("== Downloading " + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        with open(file_path, "wb") as file:
            file.write(data.read())
        file_processed = os.path.join(self.root, self.processed_folder)
        print("== Unzip from " + file_path + " to " + file_processed)
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(file_processed)
        zip_ref.close()
    print("Download finished.")
