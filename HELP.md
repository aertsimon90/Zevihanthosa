# How to Use Zevihanthosa – A Detailed Guide

## Step 1: Prepare Yourself

Before diving into Zevihanthosa, it’s important to approach it with a relaxed mindset. While Zevihanthosa is designed to be beginner-friendly, AI development always requires some patience and experimentation.

## Step 2: Get the Repository

To use Zevihanthosa, you need to download it from GitHub. The official repository is:

**https://github.com/aertsimon90/Zevihanthosa**

### Installing Git

If you haven’t installed Git yet, follow these steps:

Windows Users: Download Git by searching for "GIT Download" on the internet and install it.

Linux Users: Open a terminal and install Git using the package manager:

```bash
apk install git  # For Alpine Linux  
sudo apt install git  # For Debian-based systems  
sudo dnf install git  # For Fedora-based systems
```

Mac Users: Git usually comes pre-installed, but if not, you can install it using Homebrew:

```bash
brew install git
```

### Cloning the Repository

Once Git is installed, open a command-line interface (CLI) and navigate to your desired directory:

**Windows:**
Open Command Prompt (cmd) and navigate to the Desktop using:

```bash
cd Desktop
```

**Linux/Mac:**
Open a terminal and navigate to the desired folder where you want to download Zevihanthosa.


Now, clone the repository:

```bash
git clone https://github.com/aertsimon90/Zevihanthosa
```

This command will create a folder named "Zevihanthosa" containing all the necessary files.

## Step 3: Navigate to the Zevihanthosa Directory

Move into the downloaded Zevihanthosa directory using:

```bash
cd Zevihanthosa
```

Alternatively, Windows users can navigate to the folder using the File Explorer.

## Step 4: Importing Zevihanthosa into Your Python Code

### Understanding File Naming Conventions

Zevihanthosa model files follow this naming structure:

**zevihanthosa_ModelName_Version-VersionDetail.py**

However, when importing the model into Python, replace "-" with "_" due to Python import rules. The correct format is:

**zevihanthosa_ModelName_Version_VersionDetail.py**

Alternatively, if you want a custom import name, you can rename the file to something like:

**custom.py**

This allows you to import the file using:

```py
import custom
```

### Importing the Model

Ensure that your Python script is in the same directory as the Zevihanthosa files, or adjust the sys.path accordingly. Then, use the correct import format:

```py
import zevihanthosa_ZevihaNut_1_5 # Example for ZevihaNut/1.5
```

> Do not use from zevihanthosa_X import Y. Always use import zevihanthosa_X for full integration.

## Step 5: Exploring the Source Code

Zevihanthosa is designed to be easy to understand. To get a feel for how it works:

Open the .py files and read through the code.

Check the README.md for more details on available functions and configurations.

Experiment with simple test scripts to see how different components behave.

## Step 6: Using Basic Functions

For beginners, Zevihanthosa includes simplified functions that make it easier to work with AI concepts.

For example, to use the process_noob function, which processes input in a simple way:

```py
import zevihanthosa_ZevihaNut_1_5

brain = zevihanthosa_ZevihaNut_1_5.Brain()  
print(brain.process_noob(1))  # Example of a simple processing function
```

This allows you to quickly test and understand how Zevihanthosa processes data.

## Step 7: Staying Updated

Currently, pip support is not available, but future versions may include it. In the meantime:

Regularly check the GitHub repository for updates.

Read through release notes to understand new features and improvements.

Update your local repository using:

```bash
git pull origin main
```


---

## Summary

Zevihanthosa is a beginner-friendly AI framework that simplifies deep learning concepts while maintaining powerful customization options. By following this guide, you can:

1. Download Zevihanthosa from GitHub.


2. Set up your environment with Git and Python.


3. Import Zevihanthosa using the correct naming conventions.


4. Experiment with its functions to understand how it works.


5. Stay updated with new versions and features.



By using Zevihanthosa, even those new to AI development can start building and experimenting with artificial intelligence models without dealing with overly complex frameworks like TensorFlow or PyTorch.

