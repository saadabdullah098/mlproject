from setuptools import setup, find_packages
from typing import List

#Build a function that pulls all the required packages from requirements.txt and passes it to install_requires
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function pull all the required packages from requirements.txt and passes it to install_requires
    '''
    with open(file_path) as file_obj:
        #reads lines 
        requirements=file_obj.readlines()
        #replaces \n from end of lines
        requirements=[req.replace('\n', '') for req in requirements]
				
		#This removes the -e . when the package is being built
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

#Metadata about the entire project
setup(
    name='mlproject',
    version='0.0.1',
    author='Saad Abdullah',
    author_email='sabdullah201098@gmail.com',
    #When this is run, it looks for the app package inside src 
    packages=find_packages(),
    #This finds all the dependicies 
    install_requires=get_requirements('requirements.txt')
)