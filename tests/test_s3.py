from config import settings
import h5py
import os.path as osp
import eeghdf

print(settings.to_dict())


s3_url_endpoint = settings.GCP_S3_COMPAT_STORAGE_ENDPOINT_URL
bucket_name = settings.BUCKET_NAME
file_name = r"absence_epilepsy.eeg.h5"

h5_url = osp.join(s3_url_endpoint, f"{bucket_name}/{file_name}")
key_id = settings.GCP_S3_COMPAT_ACCESS_KEY
key_secret = settings.GCP_S3_COMPAT_SECRET_KEY
region = settings.S3_REGION


def test_basic_s3():

    h5 = h5py.File(
        h5_url,
        driver="ros3",
        aws_region=region.encode("utf-8"),
        secret_id=key_id.encode("utf-8"),
        secret_key=key_secret.encode("utf-8"),
    )

    pt = h5["patient"]
    rec = h5["record-0"]
    print(list(pt.attrs))
    print(list(rec.attrs))


vfdkwargs = dict(
    s3_bucket_name=bucket_name,
    s3_url_endpoint=s3_url_endpoint,
    access_key_id=key_id,
    secret_access_key=key_secret,
    region=region,
)

hf = eeghdf.Eeghdf(file_name, vfd="ros3", vfd_kwargs=vfdkwargs)

print(hf.rawsignals[5, 0:10])
print(hf.phys_signals[0, 0:10])

session = eeghdf.S3_session(
    s3_bucket_name=bucket_name,
    s3_url_endpoint=s3_url_endpoint,
    access_key_id=key_id,
    secret_access_key=key_secret,
    region=region,
)
f = session.get(file_name)
print(f.phys_signals[10,1000:1020])