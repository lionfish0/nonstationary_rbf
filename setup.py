from distutils.core import setup
setup(
  name = 'nonstationary_rbf',
  packages = ['nonstationary_rbf'],
  version = '1.01',
  description = 'A nonstationary RBF (EQ) kernel for GPy',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/nonstationary_rbf.git',
  download_url = 'https://github.com/lionfish0/nonstationary_rbf.git',
  keywords = ['non-stationary','kernel','gaussian process'],
  classifiers = [],
  install_requires=['GPy'],
)

