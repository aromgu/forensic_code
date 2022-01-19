
"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile
import portalocker
import requests
from tqdm import tqdm

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...'%(fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s"%url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname

_model_sha1 = {name: checksum for checksum, name in [
    # resnest
    ('fb9de5b360976e3e8bd3679d3e93c5409a5eff3c', 'resnest50'),
    # ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    # ('d7fd712f5a1fcee5b3ce176026fbb6d0d278454a', 'resnest200'),
    # ('51ae5f19032e22af4ec08e695496547acdba5ce5', 'resnest269'),
    # # rectified
    # #('9b5dc32b3b36ca1a6b41ecd4906830fc84dae8ed', 'resnet101_rt'),
    # # resnet other variants
    # ('a75c83cfc89a56a4e8ba71b14f1ec67e923787b3', 'resnet50s'),
    # ('03a0f310d6447880f1b22a83bd7d1aa7fc702c6e', 'resnet101s'),
    # ('36670e8bc2428ecd5b7db1578538e2dd23872813', 'resnet152s'),
    # # other segmentation backbones
    # ('da4785cfc837bf00ef95b52fb218feefe703011f', 'wideresnet38'),
    # ('b41562160173ee2e979b795c551d3c7143b1e5b5', 'wideresnet50'),
    # # deepten paper
    # ('1225f149519c7a0113c43a056153c1bb15468ac0', 'deepten_resnet50_minc'),
    # # segmentation resnet models
    # ('662e979de25a389f11c65e9f1df7e06c2c356381', 'fcn_resnet50s_ade'),
    # ('4de91d5922d4d3264f678b663f874da72e82db00', 'encnet_resnet50s_pcontext'),
    # ('9f27ea13d514d7010e59988341bcbd4140fcc33d', 'encnet_resnet101s_pcontext'),
    # ('07ac287cd77e53ea583f37454e17d30ce1509a4a', 'encnet_resnet50s_ade'),
    # ('3f54fa3b67bac7619cd9b3673f5c8227cf8f4718', 'encnet_resnet101s_ade'),
    # # resnest segmentation models
    # ('4aba491aaf8e4866a9c9981b210e3e3266ac1f2a', 'fcn_resnest50_ade'),
    # ('2225f09d0f40b9a168d9091652194bc35ec2a5a9', 'deeplab_resnest50_ade'),
    # ('06ca799c8cc148fe0fafb5b6d052052935aa3cc8', 'deeplab_resnest101_ade'),
    # ('7b9e7d3e6f0e2c763c7d77cad14d306c0a31fe05', 'deeplab_resnest200_ade'),
    # ('0074dd10a6e6696f6f521653fb98224e75955496', 'deeplab_resnest269_ade'),
    # ('77a2161deeb1564e8b9c41a4bb7a3f33998b00ad', 'fcn_resnest50_pcontext'),
    # ('08dccbc4f4694baab631e037a374d76d8108c61f', 'deeplab_resnest50_pcontext'),
    # ('faf5841853aae64bd965a7bdc2cdc6e7a2b5d898', 'deeplab_resnest101_pcontext'),
    # ('fe76a26551dd5dcf2d474fd37cba99d43f6e984e', 'deeplab_resnest200_pcontext'),
    # ('b661fd26c49656e01e9487cd9245babb12f37449', 'deeplab_resnest269_pcontext'),
    ]}

encoding_repo_url = 'https://s3.us-west-1.wasabisys.com/encoding'
#encoding_repo_url = 'https://hangzh.s3.amazonaws.com/'
_url_format = '{repo_url}models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.encoding', 'models')):
    r"""Return location for the pretrained on local file system.
    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.
    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if name not in _model_sha1:
        from torchvision.models.resnet import model_urls
        if name not in model_urls:
            raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
        root = os.path.expanduser(root)
        return download(model_urls[name],
                        path=root,
                        overwrite=True)
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, file_name+'.pth')
    sha1_hash = _model_sha1[name]

    lockfile = os.path.join(root, file_name + '.lock')
    with portalocker.Lock(lockfile, timeout=300):
        if os.path.exists(file_path):
            if check_sha1(file_path, sha1_hash):
                return file_path
            else:
                print('Mismatch in the content of model file {} detected.' +
                      ' Downloading again.'.format(file_path))
        else:
            print('Model file {} is not found. Downloading.'.format(file_path))

        zip_file_path = os.path.join(root, file_name+'.zip')
        repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        with zipfile.ZipFile(zip_file_path) as zf:
            zf.extractall(root)
        os.remove(zip_file_path)

        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.encoding', 'models')):
    r"""Purge all pretrained model files in local file store.
    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
