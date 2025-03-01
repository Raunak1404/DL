# HASHGRAM - Fake Meets Check
<img src="https://github.com/user-attachments/assets/dae92b6f-e3a6-456b-a594-5f4ebabcc060" width="300" alt="logo image">

## Project Goal:
The goal is to verify the authenticity of user-generated content using Multi-Modal AI detection (images, videos) and to aware online users malicious manipulation or plagiarism

### Setting Up the Virtual Environment

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```

3. Activate the virtual environment:
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
Now you're all set up to work with the project!


### To start the project - 
- clone the repository to your system
- __download the firebase-login folder__ from here: https://drive.google.com/file/d/1lxr8XdhMuQwvhFDE_4_wVX-kSeYtNEH6/view?usp=sharing
- make sure you use the VScode IDE or any preferred IDE
-  -------------------------**MAKE SURE TO DOWNLOAD BEFORE PROCEEDING WITH THE STEPS BELOW**----------------------------------------
- direct your command line tool/terminal to the __firebase-login folder__
- enter __'npm start'__ in the command line tool/terminal
- once you reach the HashGram website __sign up__ as a new user with your email address, user name and your preferred password
- once you are in, CONGRATULATIONS! you will be able to use our social media clone as a __simulation to experience how our model detects deepfake videos or copied/edited video clips (+ giving you the link to the original video) that come on your feed!__
- Coming to the main utility (create a post) page, __you can choose to upload a form of media__, and HashGram will undergo SHA-256 Hashing, deepfake detection, and then ultimately deep learning (ResNet50) to detect deepfake/unoriginal content. In the case of images also, you can check whether its original or not.
- But do you users know the best part? ALL OF THIS HAPPENS IN AN INSTANT: Our software gives you the __labels__ in the form of frames of colors: yellow and red, to see whether it is __deepfake__ or __copied/edited with the link to the video option__



<img src="https://github.com/user-attachments/assets/e070ac28-3853-4861-a88e-b3ba3e49d647" width="400">
<img src='https://github.com/user-attachments/assets/0b91205a-fa94-4979-8483-b760e6c00c18' width='400'>


<img src='https://github.com/user-attachments/assets/3164354b-0b7d-422a-bc4a-fe153c315a58' width='500'>
<img width="500" alt="image" src="https://github.com/user-attachments/assets/688d0002-d480-4faa-91e4-1072b8abbeb3" />

### Additional Notes -
- firebase-login contains the frontend code of our social media clone web app.
- youtube_videos_dataset - Sheet1.csv is a dataset created by us with 99 data values. It is used to train our ResNet50 model to find the original links of clipped/edited videos uploaded on social media.

#### Link to our google slides:
https://docs.google.com/presentation/d/1df9ASV2-7qzF7RhelTqc03VDmb7xYNoG2BvvoBnpiFw/edit?usp=sharing


