# Understanding how to use streaming S3 storage and compatible systems
- credit to Paul Nuyujukian for figuring this out and giving references to Vamsiramakrishna's mediume article

The syntax is:
```
h5 = h5py.File(<h5 url>, driver='ros3',
aws_region='<region>'.encode('utf-8'),
secret_id='<key_id>'.encode('utf-8'),
secret_key='<key_secret>'.encode('utf-8') )
```

[A study on using google cloud storage with the s3 compatibility api](https://vamsiramakrishnan.medium.com/a-study-on-using-google-cloud-storage-with-the-s3-compatibility-api-324d31b8dfeb)

Here is an example I set up with my own personal google account. This won't work for you as it is private, but you can create your own google cloud storage setup per the instructions in the article above.

starting with my personal GCP account
created project: cloudstorage-s3
created bucket: test_hdf_storage_leemesser

created service account: 

storage endpoint URL: https://storage.googleapis.com

created gcp_s3_compat_access_key and gcp_s3_compat_secret_key



The GCP_URL is a path argument to the file:
```
The GCP_URL = <storage endpoint URL>/<created bucket>/<file name>
```

### Examples of access from medium article using amazon boto3 interface

#### as a resource
```
s3_resource = boto3.resource(service_name='s3', endpoint_url=GCP_URL, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=GCP_REGION_NAME)
```

As a Session:
```
session = Session()
s3_session = session.resource(service_name='s3', endpoint_url=GCP_URL, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=ACCESS_KEY, region_name=GCP_REGION_NAME)
As a Client
s3_client = boto3.client('s3', endpoint_url= GCP_URL, aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=GCP_REGION_NAME)
```


---

### My tests

this works too, [hdf docs reference](https://www.hdfgroup.org/solutions/cloud-amazon-s3-storage-hdf5-connector/)
```
h5ls --vfd=ros3  --s3-cred="(us-west1,GOOG1EVQXL6US3D2LB4WJNY3YKPDFCJMUKLXJD4JIOATYNMU2RFCJPNCQFWRI,i7PMmdrVCOvOa/xIIoLoquzQ3B+sFgdOdz1yXRFw)" -r https://storage.googleapis.com/test_hdf_storage_leemesser/absence_epilepsy.eeg.h5
```

and test_s3.py works with python 3.8 and 3.10 with h5py 3.3 and 3.6 installed from conda-forge. (Note that the ubuntu 20.04 libhdf5 is not compiled with ros3 VFD driver support.)
```python
import h5py
import os.path as osp

# after uploading file absence_epilepsy.eeg.h5 to the bucket

bucket_name = r"test_hdf_storage_leemesser"
file_name = r"absence_epilepsy.eeg.h5"
h5_url = osp.join(h5_url_base, f"{bucket_name}/{file_name}")
key_id = gcp_s3_compat_access_key
key_secret = gcp_s3_compat_secret_key
region = "us-west1" # for example

h5 = h5py.File(h5_url, driver='ros3',
               aws_region=region.encode('utf-8'),
               secret_id=key_id.encode('utf-8'),
               secret_key=key_secret.encode('utf-8') )



"""

