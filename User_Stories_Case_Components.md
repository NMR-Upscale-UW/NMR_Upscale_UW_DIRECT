## User Stories

ML Technician 

Needs to perform periodic updates, be able to read source code, document changes, update ML models with newer more accurate models.

Technician may need access to new and original data, so that they may be able to validate a newer model or if code breaks, validate the existing model.

Tech would need to be able to collaborate effectively with other technicians but keeping "secrets" kept secret (Licensing etc?)

	* Remote work? How does code security work if not on a local machine

Be able to validate model performance etc.

Log in authetication/token

Lab Tech
---------

Patrick is a chemist who has obtained an NMR spectrum that was obtained at a 60 MHz. He needs to upscale the resolution so that the peak resolution is more resolved for analysis. 

Ideally, the NMR Upscaling software would communicate directly with the NMR hardware so that output spectra experimentally taken would be directly inputted into the ML model for upscaling. 

	* Patrick needs to know labels/selections for the spectrum (what molecule, frequency scaled to, certainty, etc.)
	* If not directly imported from NMR hardware, Patrick should have the option to input data himself

Other Software Engineers
------

Jane and John are software engineers working on implementing the code into their operating system. It is important that they know what the code does, what each function does and potential errors that could result from these functions depending on input and output, how to translate the code to a machine oriented programming language, and analyze bugs that could result from the software or its implementation.

	* Annotated code is important
	* Docstrings for functions
	* Open source/licensing 


High Level Leadership
----------------------

Heather is in a high level leadership position in the organization. It is important that she knows how much time, use, and resources are allocated to the development and maintanence of the software.

	* Output data/plots such as (usage per day, model performance and accuracy, who is logging in to track manpower)
	* Statistics and model performance, translated to something readable to someone unversed in ML or NMR


## Example Use Cases

ML Technician
* Technician receives more experimental data to input in the model
	* Prompts to input data
	* Able to validate new model performance

ML Technician
* Technician finds that a new ML model might be beneficial to the upscaling
	* Access to the code and workflow to add, train, and validate new model
	* Output successfully is incorporated with the software


Lab Tech
--------
* Patrick wants to select what upscaling method he would like to scale to
	* Prompted with selections of models trained (100, 200, 300, 400, etc.) 
		* Software tells him statistical confidence in the results for each
	* Patrick then makes the determination of what resolution he is confident and analyzes experimental results

Lab Tech
-------
* Patrick has both a 60 MHz and 400 MHz spectrum that he experimentally took
	* He wants the option to add the data to a repository for inclusion for model performance

Software Engineer
------
* The software the engineer is working on has been updated and can no longer input or output spectral data
	*  The engineer needs access to the code so they can work on changes to incorporate it in the new updates
	*  The engineer is prompted with a authentication key to open and work on the code
	*  Upon entering the key, they can access and work on changes

Component Design and Ideas
------

```nmrsim```
What it does: An open source python library that contains many functions for the design, synthetic development, and display of NMR spectra data.
Inputs: See README for more details on the functions
Outputs: Synthetic or experimental NMR data as plots or raw array data
Interactions: User/ML tech would import this library to generate or interact with NMR spectra on the local machine

```authenticate```
What it does: Prompted to make edits to the code, user inputs a user and password that would grant them access to make changes to the software
Inputs: Username and password
Outputs: Access
Interactions: User inputs a username and password

```model.py```
What it does: An output of the training of the machine learning model that can be imported to future notebooks or environments for ML predictions from experimental data
Inputs: Directory for where experimental data is stored
Outputs: A model for ML and an upscaled NMR spectra
Interactions: User can specify what model (100, 200, 300 Mhz etc)

```model.pkl```
What it does: A saved model for each upscale resolution (may need multiple models). Pretrained from model.py
Inputs: Prediction data, takes new nmr data and makes a model prediction for new data
Outputs: model prediction as an NMR spectrum
Interactions: user calls this on some interface

```import_new_data```
What it does: Allows a user to import new experimental data from a local repository
Inputs: Experimental NMR spectra (in csv format?)
Output: Numpy arrays for each new spectrum that can be input to a prediction model
Interactions: User calls this function to import their data, specificying the file that needs imported

Extra
-----
Machine Learning Technician
* Markdown, for tracking/describing file structures, functions, what things do, etc.. Good software engineering practices
* Docstrings i.e. define in detail what functions do
* Frequent comments on code
* Changelog?
	* Descriptive commit messages for our own sanity


Lab Tech (Basic User, Data In/Data Out)
* Example use case
* Manual
	* What functions, data types
	* Collection of potential error messages
	* FAQ
	* Resources/contacts, feedback mechanism/email
* Interactive UI

Resource Tracking
* Statistics for ML Performance
* Stats for usage (users, # times used, failure tracking for UI)

Communication with Outside Software

Component Design and Ideas
------

```nmrsim```
What it does: An open source python library that contains many functions for the design, synthetic development, and display of NMR spectra data.
Inputs: See README for more details on the functions
Outputs: Synthetic or experimental NMR data as plots or raw array data
Interactions: User/ML tech would import this library to generate or interact with NMR spectra on the local machine

```authenticate```
What it does: Prompted to make edits to the code, user inputs a user and password that would grant them access to make changes to the software
Inputs: Username and password
Outputs: Access
Interactions: User inputs a username and password

```model.py```
What it does: An output of the training of the machine learning model that can be imported to future notebooks or environments for ML predictions from experimental data
Inputs: Directory for where experimental data is stored
Outputs: A model for ML and an upscaled NMR spectra
Interactions: User can specify what model (100, 200, 300 Mhz etc)



