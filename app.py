import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load your model (adjust the path to your model)
MODEL_PATH = 'my_model.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_pil):
    """
    Preprocess the image to fit your model's input requirements.
    This function expects a PIL Image object as input.
    """
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    image_pil = image_pil.resize((120, 120))  # Resize to the size your model expects
    image_np = np.asarray(image_pil)
    image_np = image_np / 255.0  # Assuming your model expects pixel values in [0, 1]
    return image_np

def predict_image(image_np):
    """
    Make a prediction on a single preprocessed image and return the result.
    This function expects a numpy array as input.
    """
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    prediction = model.predict(image_np)
    predicted_class = np.argmax(prediction, axis=1)
    class_names = {0: 'No Defect', 1: 'Defect Type 1', 2: 'Defect Type 2', 3: 'Defect Type 3', 4: 'Defect Type 4'}
    result = class_names[predicted_class[0]]
    return result

def is_steel_image(image_np):
    predefined_threshold = 14  # Example threshold, needs to be tuned
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    edge_area = np.mean(edges)
    return edge_area > predefined_threshold

st.title("Defect Detection in Images")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    
    image_np = np.array(image_pil)  # Convert PIL Image to numpy array for edge detection
    if is_steel_image(image_np):
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)  # Display the PIL Image
        preprocessed_image_np = preprocess_image(image_pil)  # Preprocess the PIL Image
        if st.button('Predict'):
            result = predict_image(preprocessed_image_np)  # Make sure to pass the numpy array
            st.write(f"Prediction: {result}")
    else:
        st.error("The uploaded image does not appear to be steel. Please upload a steel image.")


# # Streamlit UI
# st.title("Defect Detection in Images")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert the uploaded file to an image
#     image = Image.open(uploaded_file)
    
#     # Display the uploaded image
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     if st.button('Predict'):
#         # Make prediction
#         result = predict_image(image)
        
#         # Display the result
#         st.write(f"Prediction: {result}")

# st.write("Note: This tool is for demonstration purposes and should not be used for critical decision-making.")


# import streamlit as st
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np

# # Load your model (this should be outside any functions to load only once)
# MODEL_PATH = 'my_model.h5'
# model = load_model(MODEL_PATH)

# def preprocess_image(image, target_size):
#     """
#     Resize the image to match the input dimensions expected by the model.
#     """
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize(target_size)
#     image = np.asarray(image)
#     image = np.expand_dims(image, axis=0)
#     return image

# def predict(image):
#     """
#     Preprocess and predict the defect on the image
#     """
#     preprocessed_image = preprocess_image(image, target_size=(120, 120))  # Adjust size as per your model
#     prediction = model.predict(preprocessed_image)
#     return prediction

# # Streamlit application layout
# st.title("Defect Detection in Images")
# st.write("Upload an image and the model will predict if it has a defect.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     if st.button('Predict'):
#         prediction = predict(image)
#         # Adjust the following line as per your model's output
#         st.write(f"Prediction: {'Defective' if prediction[0][0] > 0.5 else 'Not Defective'}")

# st.write("Note: This tool is for demonstration purposes and should not be used for critical decision-making.")





# import streamlit as st
# import pandas as pd
# import altair as alt
# import numpy as np
# from PIL import Image
# from prediction import predict_data
# import streamlit as st
# import pandas as pd
# from io import StringIO
# #Create header

# st.write("<h1 style='color: skyblue; text-align: center;'>PREDICT<span style='color:black'>.IT</span></h1>", unsafe_allow_html=True)
# # st.write(""" from insights to foresights - Predict""")
# st.write("<h4 style='color: black;'>Great tool to predict a movie’s success</h4>", unsafe_allow_html=True)
# # st.write("By shedding light on the elements that contribute to a movie's success, we help you to make predict the success of your movie. Don’t miss out on the chance to make informed decisions about which movies to produce and invest in.")

# #image
# #image = "https://www.analyticsinsight.net/wp-content/uploads/2022/01/10-Movies-on-artificial-intelligence-that-Engineers-geek-out-on.jpg"
# image = "https://cdni.iconscout.com/illustration/premium/thumb/artificial-intelligence-3454686-2918395.png"
# # st.image(image)

# col1, col2 = st.columns(2)
# # Adding the content to the first column
# with col1:
#     st.write("By shedding light on the elements that contribute to a movie's success, we help you to make predict the success of your movie.")
#     st.write("All you have to do is to:")
#     st.markdown("- Provide relevant information in the form and press the button to submit")

#     st.markdown('''
#     <style>
#     [data-testid="stMarkdownContainer"] ul{
#         padding-left:10px;
#     }
#     </style>
#     ''', unsafe_allow_html=True)

# with col2:
#     st.image(image)
# st.text(" ")
# st.text(" ")
# st.markdown(
#     """
#     <style>
#     .page-button {
#         background-color: none;
#         padding: 0.5rem 2rem;
#         border-radius: 0.5rem;
#         border: 2px solid #82CBED;
#         cursor: pointer;

#     }
#     </style>
#     """
#     , unsafe_allow_html=True
# )

# st.markdown(
#     "<a href='https://docs.google.com/document/d/17TKULd_pSmUXFowtu_QxTwh1W8-UyTSF1sF-ccNbfi0/edit' class='page-button' target='_blank' style ='margin-left: 260px; text-decoration: none; color:#000000; '>Learn more</a>",
#     unsafe_allow_html=True)
# #Bring in the data
# data = pd.read_csv('train.csv')
# st.text(" ")
# st.text(" ")
# st.text(" ")
# st.text(" ")
# st.text(" ")
# # st.text(" ")
# # st.text(" ")
# st.write("<h3 style='color: skyblue;'>THE DATA BEING USED:</h3>", unsafe_allow_html=True)

# data

# #Create and name sidebar
# st.markdown(
#     """
#     <style>
#     .css-1aumxhk {
#     background-color: #011839;
#     background-image: none;
#     color: #ffffff
# }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.sidebar.header('FILL IN YOUR INFORMATION')

# # st.sidebar.write("""#### Choose your SG bias""")
# def user_input_features():
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

    

#     if st.sidebar.button("Predict type of movie"):
#         user_data = [uploaded_file]
#         result = predict_data([user_data])

#         # Create a Pandas DataFrame to store the inputs
#         display_data = {'Variable 1': [uploaded_file],
#                      }

#         # Convert the user data to a Pandas DataFrame
#         features = pd.DataFrame(display_data)
#         # for i in range(0,10):
#         #     st.text("")

#         st.write("<h3 style='color: skyblue;'>YOUR CHOSEN VALUES:</h3>", unsafe_allow_html=True)
#         st.write(features)
#         st.write("<h3 style='color: skyblue;'>YOUR PREDICTION OUTPUT:</h3>", unsafe_allow_html=True)

#         st.write(f"**Your movie will be:** {result[0]}")
# df_user = user_input_features()
