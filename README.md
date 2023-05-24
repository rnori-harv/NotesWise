# NotesWise
**Have your notes work for you.** We leverage an LLM to take your course notes and plan an approach to help you complete your assignments or projects.

You simply upload your notes and then ask any question related to your assignment. NotesWise will use your notes to try to come up with an answer.

## How to use

### Installation
Install the dependencies using `pip install -r requirements.txt` 

### Setup
Put your notes files into the `src/` directory. They can be either text, PDF, or microsoft word files. Then, in this directory, run `streamlit run feed.py`. A web browser should open from which you can use NotesWise.


### To-Dos
- Finish File Uploading Logic such that the user does not have to manually add their files into the `src/` directory. 
- Allow multiple different file type uploads
- Extend logic to have the LLM read problem set questions.
- Have the LLM cite from where it is obtaining its answer
