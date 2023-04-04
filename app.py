# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("emotion_classifier_pipe.pkl","rb"))


# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
def main():
	st.title("Moodify-Emotion Detection App")
	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.subheader("Emotion Detection Using Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)

			# Apply Fxn Here
			prediction = predict_emotions(raw_text)
			probability = get_prediction_proba(raw_text)
			
			add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				# st.write(probability)
				proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
				# st.write(proba_df.T)
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart(c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)	



	else:
		st.subheader("About")
		info_text=st.markdown("MOODIFY is an Emotion Detection application is designed to analyze text and detect the underlying emotional content. The application uses natural language processing (NLP) techniques to identify key words and phrases, as well as nonverbal cues like punctuation and emojis, to determine the emotional tone of the text.")
		func_text=st.subheader("Functionalities:")
		info1_text=st.markdown("1. Input Text: The user can input any text they want to analyze for emotional content. The text can be entered manually or copied and pasted from another source.")
		info2_text=st.markdown("2. Emotion Detection: Once the text has been entered, the application uses NLP algorithms to analyze the content and detect the underlying emotional tone. The application can detect a wide range of emotions, including happiness, sadness, anger, fear, surprise, and more.")
		infor3_text=st.markdown("3. Emotion Score: After analyzing the text, the application provides a score for each detected emotion, indicating the strength or intensity of that emotion in the text.")
		info4_text=st.markdown("4. Visualization: The application also provides a visual representation of the emotional content, using colors or graphs to show the relative strength of each detected emotion.")
		info5_text=st.markdown("5. Real-Time Results: The user can analyze the real time emotions of the person using the web ")
		func_text=st.subheader("Limitations:")
		lim1_text=st.markdown("1. The application is designed to analyze text only, and may not be able to accurately detect emotions in other types of content, such as images or videos.")
		lim2_text=st.markdown("2. The accuracy of the emotion detection may be influenced by factors such as the quality of the text input, the complexity of the language used, and cultural differences in the interpretation of emotions.")
		func_text=st.subheader("Applications:")
		appli1_text=st.markdown("1. Customer Service: Emotion detection can be used in customer service settings to analyze customer feedback, such as emails or social media posts, to determine the emotional tone of the customer's message. This information can be used to identify customer needs and improve customer satisfaction. ")
		appli2_text=st.markdown("2. Market Research: Emotion detection can be used in market research to analyze consumer feedback on new products or services. By understanding the emotional response to a new product or service, companies can make better decisions about product development and marketing strategies.")
		appli3_text=st.markdown("3. Mental Health: Emotion detection can be used in mental health settings to identify and monitor changes in a patient's emotional state over time. This information can be used to inform treatment decisions and improve patient outcomes.")
		appli4_text=st.markdown("4. Education: Emotion detection can be used in educational settings to analyze student feedback and identify areas where students may be struggling emotionally. This information can be used to provide additional support and resources to students as needed.")
		appli5_text=st.markdown("5. Social Media: Emotion detection can be used to monitor social media platforms for trends and patterns in emotional content. This information can be used by marketers, politicians, or other groups to identify and respond to changes in public opinion or sentiment.")
		

		add_page_visited_details("About",datetime.now())






if __name__ == '__main__':
	main()