from testconfig import settings
import pytest
import h5py
import os.path as osp
import eeghdf

NOT_CONFIGURED = True
try:
    s3_url_endpoint = settings.S3_COMPAT_STORAGE_ENDPOINT_URL
    bucket_name = settings.S3_BUCKET_NAME
    file_name = r"absence_epilepsy.eeg.h5"

    full_s3_url = osp.join(s3_url_endpoint, f"{bucket_name}/{file_name}")
    key_id = settings.S3_COMPAT_ACCESS_KEY_ID
    key_secret = settings.S3_COMPAT_SECRET_ACCESS_KEY
    region = settings.S3_REGION
    NOT_CONFIGURED = False


    vfdkwargs = dict(
        s3_bucket_name=bucket_name,
        s3_url_endpoint=s3_url_endpoint,
        access_key_id=key_id,
        secret_access_key=key_secret,
        region=region,
    )

except:
    print("S3 not configured, likely need to set up an S3 bucket and create .secrets.toml")
    pass


def test_basic_s3():
    """
    Test the abilty to use h5py files with S3 for our eeghdf files
    """
    if NOT_CONFIGURED:
        pytest.skip("S3 not configured")
        
    h5 = h5py.File(
        full_s3_url,
        driver="ros3",
        aws_region=region.encode("utf-8"),
        secret_id=key_id.encode("utf-8"),
        secret_key=key_secret.encode("utf-8"),
    )

    pt = h5["patient"]
    rec = h5["record-0"]
    print(list(pt.attrs))
    print(list(rec.attrs))


def test_s3_direct_vfd():
    if NOT_CONFIGURED:
        pytest.skip("S3 not configured")

    hf = eeghdf.Eeghdf(file_name, vfd="ros3", vfd_kwargs=vfdkwargs)

    print(hf.rawsignals[5, 0:10])
    print(hf.phys_signals[0, 0:10])


# put it outside the function so I can write several tests using same session

if NOT_CONFIGURED == False:
    session = eeghdf.S3_session(
        s3_bucket_name=bucket_name,
        s3_url_endpoint=s3_url_endpoint,
        access_key_id=key_id,
        secret_access_key=key_secret,
        region=region,
    )

def test_s3_session():
    if NOT_CONFIGURED:
        pytest.skip("S3 not configured")

    f = session.get(file_name)
    print(f.phys_signals[10,1000:1020])
