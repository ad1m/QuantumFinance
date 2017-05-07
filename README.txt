README: created by Adam Lieberman, Binglun Li, and Wai Man Si
To run the program please install python 2.7.9 and the following dependencies (we also reccomend that you 
download the anaconda distribution): 

Flask 0.10.1: pip install flask==0.10.1
Flask-Moment 0.5.1: pip install flask-moment==0.5.1 Flask-Script 2.0.5: pip install flask-script==2.0.5
Jinja2 2.8: pip install jinja2==2.8
Lxml 3.6.0: pip install lxml==3.6.0
Matplotlib 1.4.3: pip install matplotlb==1.4.3
Datetime: pip install datetime
Requests 2.10.0: pip install requests==2.10.0
Pandas 0.18.1: pip install pandas==0.18.1
Pandas Datareader 0.2.1: pip install pandas-datareader==0.2.1
Numpy 1.9.2: pip install numpy==1.9.2 
Beautifulsoup4 4.4.1: pip install beautifulsoup4==4.4.1 
Scipy 0.16.1: pip install scipy==0.16.1
Joblib: pip install joblib


For easy installation of all dependencies during developement, copy and paste the following (at the bottom of the page)
into a file called dependencies.txt and then go to the terminal and type in pip install dependencies.txt. We have provided a file called dependencies.txt that has the dependencies below that you can pip as well. Once you have 
done so cd into the QuantumFinance-Master directory and type python app.py. This will then launch our application. You can then 
* Create an account
* Name your own portfolio
* Enter all the information (stock, shares, capital)
* It will give you an optimal portfolio and suggestion

If you have any questions please email Adam Lieberman at adam.justin.lieberman@gmail.com and I will be happy to 
assist you in running our application. For more detailed instructions please see our PDF in the DOC directory called
UserManual.pdf. 

The list of all dependencies on our environment are as follows:

Django==1.8.1
EasyProcess==0.2.3
Flask==0.10.1
Flask-Moment==0.5.1
Flask-Script==2.0.5
Ghost.py==0.2.3
Jinja2==2.8
MarkupSafe==0.23
MySQL-python==1.2.5
Pillow==3.4.2
Polygon2==2.0.6
PyAlgoTrade==0.17
PySide==1.2.4
PySocks==1.5.7
Pygments==2.1
Quandl==2.8.9
TTFQuery==1.0.4
Theano==0.6.0
VPython==6.10
Werkzeug==0.11.10
appdirs==1.4.0
appnope==0.1.0
astroid==1.4.4
backports-abc==0.4
backports.ssl-match-hostname==3.5.0.1
beautifulsoup4==4.4.1
blist==1.3.6
boto==2.39.0
certifi==2015.11.20.1
click==6.6
colorama==0.3.6
cv2==1.0
cvxopt==1.1.8
deap==1.0.2
decorator==4.0.6
dill==0.2.4
folium==0.2.1
fonttools==2.3
funcsigs==0.4
functools32==3.2.3.post2
future==0.15.2
geopy==1.11.0
ghost==0.4.1
gnureadline==6.3.3
gym==0.0.2
ipykernel==4.2.2
ipython==4.1.0rc2
ipython-genutils==0.1.0
ipywidgets==4.1.1
itsdangerous==0.24
joblib==0.10.3
jsonschema==2.5.1
jupyter==1.0.0
jupyter-client==4.1.1
jupyter-console==4.1.0
jupyter-core==4.0.6
lazy-object-proxy==1.2.1
lrlclib==1.0
lxml==3.6.0
matplotlib==1.4.3
mistune==0.7.1
mock==1.3.0
nbconvert==4.1.0
nbformat==4.0.1
nltk==3.0.2
nose==1.3.7
notebook==4.1.0
numpy==1.9.2
oauthlib==0.7.2
pandas==0.18.1
pandas-datareader==0.2.1
path.py==8.1.2
pbr==1.3.0
peewee==2.8.3
pexpect==4.0.1
pickleshare==0.6
plotly==1.12.2
ptyprocess==0.5.1
pyPdf==1.13
pygal==2.2.3
pylint==1.5.4
pyparsing==2.0.3
pyscreenshot==0.4.2
python-dateutil==2.4.2
python-highcharts==0.2.0
pytz==2016.4
pyxley==0.0.9
pyzmq==15.2.0
qtconsole==4.1.1
requests==2.10.0
requests-file==1.4
requests-oauthlib==0.4.2
scikit-learn==0.15.0
scikits.statsmodels==0.3.1
scipy==0.16.1
selenium==3.0.1
simplegeneric==0.8.1
simplejson==3.8.1
singledispatch==3.4.0.3
six==1.10.0
sklearn==0.0
stem==1.4.0
terminado==0.6
textblob==0.9.0
tinydb==3.2.1
tornado==4.3
traitlets==4.1.0
tweepy==3.3.0
twython==3.2.0
vboxapi==1.0
virtualenv==13.1.0
wrapt==1.10.6
wsgiref==0.1.2
wxPython==3.0.0.0
wxPython-common==3.0.0.0
yahoo-finance==1.2.1

Note, we all used Mac OS X and can only confirm that it works on Mac. It should work on windows and Linux however installing python and packages might be different. Additionally, we have tested our web application on Safari and Google Chrome. Please use one of these browsers to view the application.