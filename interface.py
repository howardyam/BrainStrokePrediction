import tkinter
from tkinter import ttk
import customtkinter
from customtkinter import CTkImage
import tkinter.font as tkFont
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,chi2
import io
import os
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score, recall_score, f1_score, roc_curve, roc_auc_score,auc
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk
from imblearn.under_sampling import EditedNearestNeighbours


FILE_PATH = 'BRFSS2021_FeatureSelection_LiteratureReview.csv'
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.feature_values = {
            'Gender': [(1, 'Male'), (2, 'Female')],
            'Heart_Disease': [(0, 'No MI or CHD'), (1, 'MI or CHD')],
            'High_Blood_Pressure': [(0, 'No'), (1, 'Yes')],
            'PhysicalActivity': [(0, 'No Activity'), (1, 'Active')],
            'High_Cholesterol': [(0, 'No'), (1, 'Yes')],
            'Stroke': [(0, 'No'), (1, 'Yes')],  # Assuming this is for displaying purpose only
            'Depression': [(0, 'No'), (1, 'Yes')],
            'BMI_Category': [(1, 'Underweight'), (2, 'Normal'), (3, 'Overweight'), (4, 'Obese')],
            'Age_Group': [(1, '18-24'), (2, '25-34'), (3, '35-44'), (4, '45-54'), (5, '55-64'), (6, '65 or older')],
            'Smoking_Status': [(0, 'Never smoke'), (1, 'Smoke')],
            'Urban_Status': [(1, 'Urban'), (2, 'Rural')],
            'Marital_Status': [(1, 'Married'), (2, 'Single'), (3, 'Unmarried couple')],
            'Vegetable_Intake': [(0, '< 1 time/day'), (1, '>= 1 time/day')],
            'Fruit_Intake': [(0, '< 1 time/day'), (1, '>= 1 time/day')],
            'Income_Group': [(1, '< $15,000'), (2, '$15,000-$24,999'), (3, '$25,000-$34,999'), (4, '$35,000-$49,999'),
                             (5, '$50,000-$99,999'), (6, '$100,000-$199,999'), (7, '>= $200,000')],
            'Education_Level': [(1, 'No High School'), (2, 'High School Grad'), (3, 'Some College'),
                                (4, 'College Grad')],
            'Race': [(1, 'White'), (2, 'Black or African American'), (3, 'American Indian or Alaskan Native'),
                     (4, 'Asian'), (5, 'Others')],
            'Diabetes': [(0, 'No'), (1, 'Yes')],
            'Current_ESmoker': [(0, 'Non-user'), (1, 'User')]
        }
        self.title("Brain Stroke Prediction And Data Visualization")
        self.geometry("1920x1080")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.setup_sidebar()
        self.setup2_experiment_page()
        self.home_page()


        self.setup3_experiment_page()
        self.show_frame(self.experiment2_page_frame)

        self.logisticregression_model = None
        self.model_performance = {}
        self.model_performance1 = {}
        self.selected_features = []
        self.text_item_counter = 0
        self.k_fold_results = {}
        self.k_fold_means = {}
        self.chi_square_results = {}
        self.roc_data = {}
        self.show_frame1(self.experiment2_page_frame)

    def setup_sidebar(self):
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nswe")

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Brain Stroke\nPrediction &\nData Visualization", font=customtkinter.CTkFont(size=16, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)

        self.sidebar_button_experiment2 = customtkinter.CTkButton(self.sidebar_frame, text="Data Preprocessing",
                                                                  command=lambda: self.show_frame(
                                                                      self.experiment2_page_frame))
        self.sidebar_button_experiment2.pack(pady=10, padx=20)

        self.sidebar_button_experiment3 = customtkinter.CTkButton(self.sidebar_frame, text="Model Evaluation",
                                                                  command=lambda: self.show_frame(
                                                                      self.experiment3_page_frame))
        self.sidebar_button_experiment3.pack(pady=10, padx=20)

        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Brain Stroke Prediction",
                                                        command=lambda: self.show_frame(self.main_page_frame))
        self.sidebar_button_1.pack(pady=10, padx=20)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Mode:", anchor="w")
        self.appearance_mode_label.pack(padx=20, pady=(10, 0))  # Use pack instead of grid
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Dark","Light"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.pack(padx=20, pady=(10, 10))



    def home_page(self):
        self.main_page_frame = customtkinter.CTkFrame(self)
        self.main_page_frame.grid(row=0, column=1, sticky="nsew")
        self.main_page_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.main_page_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13), weight=1)

        welcome_title = customtkinter.CTkLabel(self.main_page_frame, text="Welcome to Brain Stroke Prediction Interface",
                                               font=("Helvetica", 30, "bold"),
                                               wraplength=1000,)
        welcome_title.grid(row=0, column=0, columnspan=2, pady=20, padx=20, sticky="w")

        description_label = customtkinter.CTkLabel(self.main_page_frame,
                                                   text="You can assess the risk of getting brain stroke through risk factors of stroke by picking the risk factor you have below.",
                                                   font=("Helvetica", 20, "bold"),
                                                   wraplength=750,
                                                   justify=tkinter.LEFT)
        description_label.grid(row=1, column=0, columnspan=2, pady=0, padx=20, sticky="w" )





    def setup2_experiment_page(self):
        self.experiment2_page_frame = customtkinter.CTkFrame(self)
        self.experiment2_page_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.experiment2_page_frame.grid_columnconfigure((0, 1, 2,3,4), weight=1)
        self.experiment2_page_frame.grid_rowconfigure((0, 1, 2, 3), weight=1)

        title_label = customtkinter.CTkLabel(self.experiment2_page_frame, text="Data Preprocessing Dashboard", font=("Helvetica", 24, "bold"))
        title_label.grid(row=0, column=0, columnspan=6, sticky="w", pady=15, padx=10)

        """
         #Button 1
        button_1 = customtkinter.CTkButton(self.experiment2_page_frame, text="Button 1",command=self.load_clean_and_display)  # Add command if needed
        button_1.grid(row=1, column=0, padx=20, pady=10, sticky="w")



        # Create CTkScrollableFrame and place a Label inside it for displaying data
        self.widget_1 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame)
        self.widget_1.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
         #Title for data inside widget_1
        title_widget_1 = customtkinter.CTkLabel(self.widget_1, text="Load Data", font=("Helvetica", 12, "bold"),
                                                anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_1.pack(fill="both", padx=10, pady=5)
        self.label_1 = customtkinter.CTkLabel(self.widget_1, text="", justify=tkinter.LEFT,
                                           wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 8, "bold"))
        self.label_1.pack(fill="both", padx=10, pady=5)


        self.widget_2 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame)
        self.widget_2.grid(row=1, column=2, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_2 = customtkinter.CTkLabel(self.widget_2, text="Clean Data", font=("Helvetica", 12, "bold"),
                                                anchor="nw", fg_color='transparent',text_color="#3a7ebf")
        title_widget_2.pack(fill="both", padx=10, pady=5)
        self.label_2 = customtkinter.CTkLabel(self.widget_2, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              , font=("Helvetica", 12, "bold"))
        self.label_2.pack(fill="both", expand=True)

        self.widget_3 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame)
        self.widget_3.grid(row=1, column=3, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_3 = customtkinter.CTkLabel(self.widget_3, text="Impute Data", font=("Helvetica", 12, "bold"),
                                                anchor="nw", fg_color='transparent',text_color="#3a7ebf")
        title_widget_3.pack(fill="both", padx=10, pady=5)
        self.label_3 = customtkinter.CTkLabel(self.widget_3, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 12, "bold"))
        self.label_3.pack(fill="both", expand=True)
        """


        self.widget_5 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,
                                                         orientation="horizontal",scrollbar_button_color="#333333",
                                                                scrollbar_button_hover_color="#333333")
        self.widget_5.grid(row=3, column=0, columnspan=3, padx=(20, 10), pady=(10, 10), sticky="nsew")

        # Title for data inside widget_5
        title_widget_5 = customtkinter.CTkLabel(self.widget_5, text="Chi-Square Test Results",
                                                font=("Helvetica", 12, "bold"), anchor="nw",
                                                fg_color='transparent',text_color="#3a7ebf")
        title_widget_5.pack(fill="both", padx=10, pady=5)






        self.widget_7 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,
                                                         scrollbar_button_color="#333333",
                                                         scrollbar_button_hover_color="#333333")
        self.widget_7.grid(row=3, column=3, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_7 = customtkinter.CTkLabel(self.widget_7, text="Top 5 Cramer's Value", font=("Helvetica", 12, "bold"),
                                                anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_7.pack(fill="both", padx=10, pady=5)
        self.label_7 = customtkinter.CTkLabel(self.widget_7, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 12, "bold"))
        self.label_7.pack(fill="both", expand=True)



        self.widget_8 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,
                                                         scrollbar_button_color="#333333",
                                                         scrollbar_button_hover_color="#333333")
        self.widget_8.grid(row=3, column=4,padx=(20, 10), pady=(10, 10), sticky="nsew")  # Assuming 4 columns```
        title_widget_8 = customtkinter.CTkLabel(self.widget_8, text="Final Selected Features", font=("Helvetica", 12, "bold"),
                                                anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_8.pack(fill="both", padx=10, pady=5)
        self.label_8 = customtkinter.CTkLabel(self.widget_8, text="", justify=tkinter.CENTER,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 12, "bold"))
        self.label_8.pack(fill="both", expand=True)



        self.pie_chart_label = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,
                                                                scrollbar_button_color="#333333",
                                                                scrollbar_button_hover_color="#333333")
        self.pie_chart_label.grid(row=1,column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        title_widget_pie = customtkinter.CTkLabel(self.pie_chart_label, text="Distribution of Stroke ",
                                                font=("Helvetica", 12, "bold"), anchor="nw",
                                                fg_color='transparent', text_color="#3a7ebf",justify=tkinter.LEFT)
        title_widget_pie.pack(fill="both", padx=10, pady=5)
        self.graph_pie = customtkinter.CTkLabel(self.pie_chart_label, text="", fg_color="transparent")
        self.graph_pie.pack(fill="both", expand=True)

        self.pie_chart_label2 = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,scrollbar_button_color="#333333",
                                                                scrollbar_button_hover_color="#333333")
        self.pie_chart_label2.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        title_widget2_pie = customtkinter.CTkLabel(self.pie_chart_label2,
                                                  text="Distribution of Stroke \nafter Stratified Sapmling",
                                                  font=("Helvetica", 12, "bold"), anchor="nw",
                                                  fg_color='transparent', text_color="#3a7ebf", justify=tkinter.LEFT)
        title_widget2_pie.pack(fill="both", padx=10, pady=5)
        self.graph_pie2 = customtkinter.CTkLabel(self.pie_chart_label2, text="", fg_color="transparent")
        self.graph_pie2.pack(fill="both", expand=True)







        self.widget_graph = customtkinter.CTkScrollableFrame(self.experiment2_page_frame,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333"
                                                             )
        self.widget_graph.grid(row=1, column=3, rowspan=2, columnspan=2, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_graph = customtkinter.CTkLabel(self.widget_graph, text="Stacked Bar Chart For Data Comparison\nBefore and After Stratified Sampling",
                                                    font=("Helvetica",12, "bold"), anchor="nw",
                                                    fg_color='transparent', text_color="#3a7ebf", justify=tkinter.LEFT)
        title_widget_graph.grid(row=0, column=0, padx=10, pady=5, sticky="w")


        # OptionMenu for selecting the variable to visualize
        variable_names = [name for name in self.feature_values.keys()]
        self.variable_optionmenu_graph = customtkinter.CTkOptionMenu(self.widget_graph,
                                                                     values=variable_names,
                                                                     command=self.dynamic_samplingprop)
        self.variable_optionmenu_graph.grid(row=0, column=1, padx=20, pady=20, sticky="e")

        self.graph_labelDP = customtkinter.CTkLabel(self.widget_graph, text="")
        self.graph_labelDP.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        default_image_path = 'Gender_stacked_bar_chart.png'  # Ensure this path points to an actual image file
        self.display_image_in_DPlabel(default_image_path, self.graph_labelDP)
        self.widget_graph.grid_columnconfigure(0, weight=1)
        self.widget_graph.grid_columnconfigure(1, weight=1)
        self.widget_graph.grid_rowconfigure(1, weight=1)

        """
        self.test = customtkinter.CTkScrollableFrame(self.experiment2_page_frame)
        self.test.grid(row=1, column=1, rowspan=2, columnspan=2,  padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_pie = customtkinter.CTkLabel(self.test, text="Distribution of Stroke ",
                                                  font=("Helvetica", 12, "bold"), anchor="nw",
                                                  fg_color='transparent', text_color="#3a7ebf", justify=tkinter.LEFT)
        title_widget_pie.pack(fill="both", padx=10, pady=5)
        self.test1 = customtkinter.CTkLabel(self.test, text="", fg_color="transparent")
        self.test1.pack(fill="both", expand=True)
        self.graph_pie3 = customtkinter.CTkLabel(self.test, text="", fg_color="transparent")
        self.graph_pie3.pack(fill="both", expand=True)
       """

    def setup3_experiment_page(self):
        self.experiment3_page_frame = customtkinter.CTkFrame(self)
        self.experiment3_page_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.experiment3_page_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.experiment3_page_frame.grid_rowconfigure((0, 1, 2, 3,4), weight=1)
        # Title label
        title_label = customtkinter.CTkLabel(self.experiment3_page_frame, text="Model Training Dashboard",
                                             font=("Helvetica", 24, "bold"), height=100)
        title_label.grid(row=0, column=0, columnspan=5, pady=0, padx=10,sticky="w")
        title_label2 = customtkinter.CTkLabel(self.experiment3_page_frame, text="",
                                             font=("Helvetica", 24, "bold"))
        title_label2.grid(row=0, column=0, rowspan=5, pady=0, padx=10, sticky="w")
        """
        self.train_all_models_button = customtkinter.CTkButton(self.experiment3_page_frame, text="Train All Models",
                                                               command=self.train_all_models)
        #self.train_all_models_button.grid(row=1, column=0,  padx=20, pady=10, sticky="we")

        for i in range(5):  # Assuming you have 6 rows; adjust as necessary
            self.experiment3_page_frame.rowconfigure(i, weight=1)
        for j in range(6):  # Assuming 6 columns; adjust as necessary
            self.experiment3_page_frame.columnconfigure(j, weight=1)
        """
        """
        self.widget_11 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,height=125,width=250,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.widget_11.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_11 = customtkinter.CTkLabel(self.widget_11, text="Decision Tree",
                                                font=("Helvetica", 14, "bold"),
                                                anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_11.pack(fill="both", padx=10, pady=5)
        self.label_11 = customtkinter.CTkLabel(self.widget_11, text="", justify=tkinter.RIGHT,
                                               wraplength=200,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 10, "bold"))
        self.label_11.pack(fill="both", expand=True)

        self.widget_12 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,height=125,width=250,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.widget_12.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_12 = customtkinter.CTkLabel(self.widget_12, text="Logistic Regression",
                                                 font=("Helvetica", 14, "bold"),
                                                 anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_12.pack(fill="both", padx=10, pady=5)
        self.label_12 = customtkinter.CTkLabel(self.widget_12, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 10, "bold"))
        self.label_12.pack(fill="both", expand=True)

        self.widget_13 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,height=125,width=250,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.widget_13.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_13 = customtkinter.CTkLabel(self.widget_13, text="Random Forest",
                                                 font=("Helvetica", 14, "bold"),
                                                 anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_13.pack(fill="both", padx=10, pady=5)
        self.label_13 = customtkinter.CTkLabel(self.widget_13, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 10, "bold"))
        self.label_13.pack(fill="both", expand=True)

        self.widget_14 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,height=125,width=250,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.widget_14.grid(row=4, column=0, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_14 = customtkinter.CTkLabel(self.widget_14, text="KNN",
                                                 font=("Helvetica", 14, "bold"),
                                                 anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_14.pack(fill="both", padx=10, pady=5)
        self.label_14 = customtkinter.CTkLabel(self.widget_14, text="", justify=tkinter.RIGHT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 10, "bold"))
        self.label_14.pack(fill="both", expand=True)


        self.graph_option_frame = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,width=500,height=250)
        self.graph_option_frame.grid(row=1, column=1, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.graph_display_frame = customtkinter.CTkScrollableFrame(self.experiment3_page_frame,width=500,height=250)
        self.graph_display_frame.grid(row=3, column=1, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.graph_type_option_menu = customtkinter.CTkOptionMenu(self.graph_display_frame,
                                                                  values=["K-Fold Results", "ROC Comparison"],
                                                                  command=self.update_graph_display)
        self.graph_type_option_menu.pack(padx=5, pady=0)
        # Option menu to select graph type
        self.graph_option_menu = customtkinter.CTkOptionMenu(self.graph_option_frame,
                                                         values=["Overall Confusion Matrix",
                                                                 "Decision Tree ROC Curve",
                                                                 "Logistic Regression ROC Curve",
                                                                 "Random Forest ROC Curve",
                                                                 "KNN ROC Curve",
                                                                 "Overall Model Comparison",
                                                                 "Model Performance Metrics" ],

                                                         command=self.display_graph_in_option_frame)
        self.graph_option_menu.pack(padx=5, pady=0)


        self.graph_label = customtkinter.CTkLabel(self.graph_display_frame, text="",fg_color="transparent")
        self.graph_label.pack(fill="both", expand=True)

        self.graph_label1 = customtkinter.CTkLabel(self.graph_option_frame, text="",fg_color="transparent")
        self.graph_label1.pack(fill="both", expand=True)
        """

        """
        self.k_fold_results_frame = customtkinter.CTkScrollableFrame(self.experiment3_page_frame)
        self.k_fold_results_frame.grid(row=1, column=4, rowspan=2, padx=(20, 10), pady=(10, 10), sticky="nsew")
        title_widget_kfold = customtkinter.CTkLabel(self.k_fold_results_frame, text="KFold",
                                                font=("Helvetica", 14, "bold"),
                                                 anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_kfold.pack(fill="both", padx=10, pady=5)
        self.k_fold_results_label = customtkinter.CTkLabel(self.k_fold_results_frame, text="", justify=tkinter.LEFT,
                                               wraplength=400,
                                              width=100,
                                              text_color="White"
                                              ,font=("Helvetica", 12, "bold"))
        self.k_fold_results_label.pack(fill="both", expand=True)
        """

        self.graph_option_frame2 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame, width=550, height=300,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.graph_option_frame2.grid(row=1, column=2, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        title_widget_gop2 = customtkinter.CTkLabel(self.graph_option_frame2, text="10 K-fold Cross Validation",
                                                 font=("Helvetica", 14, "bold"),
                                                 anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_gop2.pack(fill="both", padx=10, pady=5)

        self.graph_display_frame2 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame, width=550, height=300,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.graph_display_frame2.grid(row=3, column=2, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        title_widget_gdp2 = customtkinter.CTkLabel(self.graph_display_frame2, text="Overall Model Performance",
                                                   font=("Helvetica", 14, "bold"),
                                                   anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_gdp2.pack(fill="both", padx=10, pady=5)

        self.graph_option_frame3 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame, width=550, height=300,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.graph_option_frame3.grid(row=1, column=0, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        title_widget_gop3 = customtkinter.CTkLabel(self.graph_option_frame3, text="Confusion Matrix",
                                                   font=("Helvetica", 14, "bold"),
                                                   anchor="nw", fg_color='transparent', text_color="#3a7ebf")
        title_widget_gop3.pack(fill="both", padx=10, pady=5)

        self.graph_display_frame3 = customtkinter.CTkScrollableFrame(self.experiment3_page_frame, width=550, height=300,
                                                             scrollbar_button_color="#333333",
                                                             scrollbar_button_hover_color="#333333")
        self.graph_display_frame3.grid(row=3, column=0, columnspan=2, rowspan=2, padx=(5, 5), pady=(5, 5),
                                       sticky="nsew")

        title_widget_gdp3 = customtkinter.CTkLabel(self.graph_display_frame3, text="ROC_AUC Graph",
                                                   font=("Helvetica", 14, "bold"),
                                                   anchor="nw", fg_color='transparent', text_color="#3a7ebf")

        title_widget_gdp3.pack(fill="both", padx=10, pady=5)

    def load_and_display_data(self):
        global dfs
        dfs = pd.read_csv(FILE_PATH)
        # Capture the output of dfs.info() to a string
        buffer = io.StringIO()
        dfs.info(buf=buffer)
        info_str = buffer.getvalue()

        loaddata= (f"DataFrame Loaded:\n{info_str}")
        print(loaddata)
        return f"DataFrame Loaded:\n{info_str}"

    def clean_data(self):
        global dfs
        dfs['Heart_Disease'] = dfs['Heart_Disease'].replace({2: 0})
        dfs['High_Blood_Pressure'] = dfs['High_Blood_Pressure'].replace({1: 0, 2: 1, 9: np.nan})
        dfs['PhysicalActivity'] = dfs['PhysicalActivity'].replace({2: 0, 9: np.nan})
        dfs['High_Cholesterol'] = dfs['High_Cholesterol'].replace({1: 0, 2: 1, 9: np.nan})
        dfs['Stroke'] = dfs['Stroke'].replace({2: 0})
        dfs['Depression'] = dfs['Depression'].replace({2: 0, 7: np.nan, 9: np.nan})
        dfs['Smoking_Status'] = dfs['Smoking_Status'].replace({2: 1, 3: 1, 4: 0, 9: np.nan})
        dfs['Marital_Status'] = dfs['Marital_Status'].replace({3: 2, 4: 2, 5: 2, 6: 3, 9: np.nan})
        dfs['Vegetable_Intake'] = dfs['Vegetable_Intake'].replace({2: 0, 9: np.nan})
        dfs['Fruit_Intake'] = dfs['Fruit_Intake'].replace({2: 0, 9: np.nan})
        dfs['Income_Group'] = dfs['Income_Group'].replace({9: np.nan})
        dfs['Education_Level'] = dfs['Education_Level'].replace({9: np.nan})
        dfs['Race'] = dfs['Race'].replace({6: 5, 7: 5, 77: np.nan, 99: np.nan})
        dfs['Diabetes'] = dfs['Diabetes'].replace({2: 0, 3: 0, 4: 0, 7: np.nan, 9: np.nan})
        dfs['Current_ESmoker'] = dfs['Current_ESmoker'].replace({1: 0, 2: 1, 9: np.nan})

        null_values_after_cleaning = dfs.isnull().sum().to_string()

        summary = "Null Values After Cleaning:\n" + null_values_after_cleaning + "\n\nData cleaned successfully.\n"
        summary += f"Remaining NaNs: {dfs.isna().sum().sum()}\n"
        printsummary = summary
        print(printsummary)
        return summary


    def impute_data(self):
        global dfs
        # Impute missing values here
        Impute_columns = ['Gender', 'Heart_Disease', 'High_Blood_Pressure', 'PhysicalActivity', 'High_Cholesterol',
                          'Stroke', 'Depression', 'BMI_Category', 'Age_Group', 'Smoking_Status', 'Urban_Status',
                          'Marital_Status', 'Vegetable_Intake', 'Fruit_Intake', 'Income_Group', 'Education_Level',
                          'Race', 'Diabetes', 'Current_ESmoker']
        for column in Impute_columns:
            if column in dfs.columns:
                dfs[column] = dfs[column].fillna(dfs[column].mode()[0])




        # Display null values after imputation
        null_values_after_imputation = dfs.isnull().sum().to_string()

        summary = "Null Values After Imputation:\n" + null_values_after_imputation + "\n"
        summary += "Missing values imputed.\n"
        summary += f"Remaining NaNs after imputation: {dfs.isna().sum().sum()}\n"
        printsummary = summary
        print(printsummary)
        return summary


    def perform_stratified_sampling(self):
        global dfs  # Ensure dfs is the full dataset
        global balanced_dfs_global
        if dfs.empty:
            print("Data not loaded. Please load data first.\n")
            return
        target_yes_count =1721
        target_no_count = 42005
        stroke_yes = dfs[dfs['Stroke'] == 1].copy()
        stroke_no = dfs[dfs['Stroke'] == 0].copy()
        if len(stroke_yes) > target_yes_count:
            sampled_yes = stroke_yes.sample(n=target_yes_count, random_state=90)
            yes_sampling_message = f"'Yes' Stroke Data After Sampling: {sampled_yes.shape}"
        else:
            sampled_yes = stroke_yes
            yes_sampling_message = "No 'Yes' sampling performed as count is within target."
        # Stratify 'no' instances if needed
        if len(stroke_no) > target_no_count:
            sampled_no = stroke_no.sample(n=target_no_count, random_state=90)
            no_sampling_message = f"'No' Stroke Data After Sampling: {sampled_no.shape}"
        else:
            sampled_no = stroke_no
            no_sampling_message = "No 'No' sampling performed as count is within target."
        # Combine sampled 'yes' and 'no' instances
        balanced_dfs = pd.concat([sampled_yes, sampled_no], ignore_index=True)
        balanced_dfs_global = balanced_dfs


    def perform_chi_square_test(self):
        global balanced_dfs_global
        if balanced_dfs_global.empty:
            self.results_text_widget_data_preprocessing.insert(tkinter.END,
                                                               "Stratified dataset is empty. Please perform stratified sampling first.\n")
            return []
        target_variable = 'Stroke'
        parameters = ['Gender', 'Heart_Disease', 'High_Blood_Pressure', 'PhysicalActivity',
                      'High_Cholesterol', 'Depression', 'BMI_Category', 'Age_Group',
                      'Smoking_Status', 'Urban_Status', 'Marital_Status', 'Vegetable_Intake',
                      'Fruit_Intake', 'Income_Group', 'Education_Level', 'Race',
                      'Diabetes', 'Current_ESmoker']
        chi_square_result = {}
        chi_square_result1 = {}
        selected_features = []
        output_lines = []
        for variable in parameters:
            contingency_table = pd.crosstab(balanced_dfs_global[target_variable], balanced_dfs_global[variable])
            chi2_stat, p, dof, expected = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()  # Total observations
            alpha = 0.05  # Significance level
            critical_value = chi2.ppf(1 - alpha, dof)
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            result = f"{variable}: Chi2={chi2_stat:.2f}, p-value={p:.2f}, Cramer's V={cramers_v:.2f} , Critical Value={critical_value:.2f}"
            output_lines.append(result)
            results = "Reject Null Hypothesis" if chi2_stat > critical_value and p < 0.05 else "Fail to Reject Null Hypothesis"
            chi_square_result1[variable] = (chi2_stat, p, dof, cramers_v, results)
            chi_square_result[variable] = {
               'Chi2 Statistic': chi2_stat,
                'p-value': p,
                'Degrees of Freedom': dof,
                'Cramér\'s V': cramers_v,
                'Critical Value': critical_value,
                'Result': 'Reject Null Hypothesis' if chi2_stat > critical_value and p < 0.05  else 'Fail to Reject Null Hypothesis'
            }
            if chi2_stat > critical_value and p < 0.05 and cramers_v > 0.10 :
                selected_features.append(variable)
        self.create_dynamic_ui_elements()
        self.selected_features = selected_features
        self.chi_square_results = chi_square_result1
        formatted_features = "\n\n".join(selected_features)
        final_selectedfeatures = f"{formatted_features}"
        # Update the label with the formatted string
        self.label_8.configure(text=final_selectedfeatures)  # Display in widget_9
        # Display chi-square test results in the ScrolledText widget
        chi_square_results_df = pd.DataFrame.from_dict(chi_square_result, orient='index').sort_values(by='Cramér\'s V',
                                                                                          ascending=False)
        chi_square_results_df['p-value'] = chi_square_results_df['p-value'].round(6)
        results_display = chi_square_results_df.to_string()
        final_chisquare = results_display
        #self.label_5.config(text=final_chisquare)  # Display in widget_5

        top5 = chi_square_results_df.head(5)
        top5_display = top5[['Cramér\'s V']].applymap(lambda x: '{:.2f}'.format(x)).to_string(header=False)
        self.label_7.configure(text=f"Top 5 Variables by Cramér's V:\n{top5_display}")
        return selected_features


    def train_decision_tree(self):
        # Ensure the global dataset variable is accessi![](../../Desktop/Screenshot_1701926899.png)ble
        global balanced_dfs_global

        if balanced_dfs_global.empty:
            print( "Stratified dataset is empty. Please perform stratified sampling and cleaning first.\n")
            return
        selected_features = self.perform_chi_square_test()
        if not selected_features:  # If no features were selected
            print("No features selected by chi-square test.\n")
            return
        # Prepare the data
        X = balanced_dfs_global[selected_features]
        y = balanced_dfs_global['Stroke']
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        unique1, counts1 = np.unique(y_train, return_counts=True)
        print(f"Class distribution before SMOTE: {dict(zip(unique1, counts1))}")
        # ////////////////////////////////////////////////////Smote
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Check new class distribution
        unique, counts = np.unique(y_train_smote, return_counts=True)
        print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

        # Initialize the Decision Tree Classifier
        clf = DecisionTreeClassifier(random_state=42)

        # Perform k-fold cross-validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_smote, y_train_smote, cv=skf, scoring='accuracy')

        self.k_fold_results['Decision Tree'] = cv_scores

        # Training on the resampled training set
        clf.fit(X_train_smote, y_train_smote)
        # ////////////////////////////////////////////////////////
        #////////////////////////////////////////////////////Original
        # Initialize the Decision Tree Classifier
        #clf = DecisionTreeClassifier(random_state=42)

        # Perform k-fold cross-validation
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(clf, X_train, y_train, cv=skf,
        #                            scoring='accuracy')  # Can use other scoring metrics as needed

        #self.k_fold_results['Decision Tree'] = cv_scores

        # Training on the training set
        #clf.fit(X_train, y_train)
        #////////////////////////////////////////////////////////

        #////////////////enn//////////////////////////////////
        # Apply ENN
        #enn = EditedNearestNeighbours()
        #X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

        # Check how the data has been balanced
        #unique, counts = np.unique(y_train_enn, return_counts=True)
        #print(f"Balanced class counts after ENN: {dict(zip(unique, counts))}")

        # Initialize the Decision Tree Classifier
        #clf = DecisionTreeClassifier(random_state=42)

        # Perform k-fold cross-validation
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(clf, X_train_enn, y_train_enn, cv=skf, scoring='accuracy')

        #self.k_fold_results['Decision Tree'] = cv_scores

        # Training on the resampled training set
        #clf.fit(X_train_enn, y_train_enn)
        # ////////////////////////////////////////////////////////
        # Predict on the test set
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        self.plot_confusion_matrix(y_test, y_pred, ['No Stroke', 'Stroke'], 'Decision Tree')

        self.model_performance['Decision Tree'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            # Include other metrics here as needed
        }
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        self.roc_data['Decision Tree'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)  # Calculate ROC AUC
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store metrics in the model_performance dictionary
        self.model_performance1['Decision Tree'] = {
             'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC_AUC': roc_auc
        }
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Decision Tree (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')

        plt.savefig("decision_tree_roc_curve.png")
        plt.close()  # Close the plot to free up memory

        # Example: Making a prediction with sample input
        # Ensure the sample input has the correct shape
        #sample_input = pd.DataFrame([[1,1,1]], columns=selected_features)  # Adjust this based on your actual feature set
        #prediction = clf.predict(sample_input)
        #prediction_proba = clf.predict_proba(sample_input)


        #    self.results_text_widget_model_training.insert(tkinter.END, f"Fold {i}: {score:.4f}\n")
        cv_scores_mean = cv_scores.mean()
        self.k_fold_means['Decision Tree'] = cv_scores.mean()

        #self.results_text_widget_model_training.insert(tkinter.END, f"5-Fold CV Score: {cv_scores_mean:.4f}\n\n")
        result_message = f"Decision Tree Model Training Completed.\n\nAccuracy: {accuracy}\n\nROC AUC: {roc_auc}\n\n"
        result_message += f"Confusion Matrix:\n\n{conf_matrix}\n\n"
        result_message += f"Classification Report:\n\n{classification_rep}"



        """
        result = (
            f"10-Fold CV Average Accuracy:  {cv_scores_mean:.4f}\n\n"
            f"Accuracy:  {accuracy:.4f}\n\n "
            f"ROC:  {roc_auc:.4f}\n\n"
            f"Classification_rep:\n{classification_rep}\n"

        )
        """
        result = (
            f"Classification_rep:\n{classification_rep}\n"
        )
        result1 = result
        print(f"This is for decision tree, {result1}")
        return result





    def train_logistic_regression(self):
        global balanced_dfs_global

        if balanced_dfs_global.empty:
            print( "Stratified dataset is empty. Please perform stratified sampling and cleaning first.\n")
            return

        selected_features = self.perform_chi_square_test()
        if not selected_features:  # If no features were selected
            print( "No features selected by chi-square test.\n")
            return

            # Display the selected features again in the model training results widget

        # Prepare the data
        X = balanced_dfs_global[selected_features]
        y = balanced_dfs_global['Stroke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        unique1, counts1 = np.unique(y_train, return_counts=True)
        print(f"Class distribution before SMOTE: {dict(zip(unique1, counts1))}")
        #////////////////Smote
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Check new class distribution
        unique, counts = np.unique(y_train_smote, return_counts=True)
        print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

        logreg = LogisticRegression(random_state=42, solver='liblinear')

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(logreg, X_train_smote, y_train_smote, cv=skf, scoring='accuracy') # Can use other scoring metrics as needed
        self.k_fold_results['Logistic Regression'] = cv_scores
        logreg.fit(X_train_smote, y_train_smote)
        #/////////////////////////////////////



        #///////////////////original
        #logreg = LogisticRegression(random_state=42, solver='liblinear')

        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(logreg, X_train, y_train, cv=skf,
        #                            scoring='accuracy')  # Can use other scoring metrics as needed
        #self.k_fold_results['Logistic Regression'] = cv_scores
        #logreg.fit(X_train, y_train)
        #///////////////////////////

        #///////////////enn///////
        #enn = EditedNearestNeighbours()
        #X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

        # Check how the data has been balanced
        #unique, counts = np.unique(y_train_enn, return_counts=True)
        #print(f"Balanced class counts after ENN: {dict(zip(unique, counts))}")

        #logreg = LogisticRegression(random_state=42, solver='liblinear')

        # Perform k-fold cross-validation
        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(logreg, X_train_enn, y_train_enn, cv=skf, scoring='accuracy')


        #self.k_fold_results['Logistic Regression'] = cv_scores
        #logreg.fit(X_train_enn, y_train_enn)

        #/////////////////////////////
        y_pred = logreg.predict(X_test)
        y_pred_proba = logreg.predict_proba(X_test)[:, 1]

        self.plot_confusion_matrix(y_test, y_pred, ['No Stroke', 'Stroke'], 'Logistic Regression')

        self.model_performance['Logistic Regression'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            # Include other metrics here as needed
        }

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        self.roc_data['Logistic Regression'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store metrics in the model_performance dictionary
        self.model_performance1['Logistic Regression'] = {
             'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC_AUC': roc_auc
        }
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Logistic Regression (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Logistic Regression')
        plt.legend(loc='lower right')
        plt.savefig("logistic_regression_roc_curve.png")
        plt.close()


        cv_scores_mean = cv_scores.mean()
        self.k_fold_means['Logistic Regression'] = cv_scores.mean()
        self.logisticregression_mode = logreg
        result = (
            f"Classification_rep:\n{classification_rep}\n"
        )
        result1 = result
        print(f"This is for Logistic regression, {result1}")
        return result


    def train_random_forest(self):
        global balanced_dfs_global

        if balanced_dfs_global.empty:
            print("Stratified dataset is empty. Please perform stratified sampling and cleaning first.\n")
            return

        selected_features = self.perform_chi_square_test()
        if not selected_features:  # If no features were selected
            print("No features selected by chi-square test.\n")
            return



        # Prepare the data
        X = balanced_dfs_global[selected_features]
        y = balanced_dfs_global['Stroke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        unique1, counts1 = np.unique(y_train, return_counts=True)
        print(f"Class distribution before SMOTE: {dict(zip(unique1, counts1))}")
        #////////////Smote
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Check new class distribution
        unique, counts = np.unique(y_train_smote, return_counts=True)
        print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

        rf = RandomForestClassifier(n_estimators=25, random_state=42)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=skf, scoring='accuracy')
        # Can use other scoring metrics as needed
        self.k_fold_results['Random Forest'] = cv_scores
        rf.fit(X_train_smote, y_train_smote)
        #///////////////////////////


        #rf = RandomForestClassifier(n_estimators=100, random_state=42)

        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(rf, X_train, y_train, cv=skf,
         #                           scoring='accuracy')  # Can use other scoring metrics as needed
        #self.k_fold_results['Random Forest'] = cv_scores
        #rf.fit(X_train, y_train)


        #/////////ENN
        #enn = EditedNearestNeighbours()
        #X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

        # Check how the data has been balanced
        #unique, counts = np.unique(y_train_enn, return_counts=True)
        #print(f"Balanced class counts after ENN: {dict(zip(unique, counts))}")

        #rf = RandomForestClassifier(n_estimators=25, random_state=42)

        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(rf, X_train_enn, y_train_enn, cv=skf, scoring='accuracy')
        # Can use other scoring metrics as needed
        #self.k_fold_results['Random Forest'] = cv_scores
        #rf.fit(X_train_enn, y_train_enn)
        #////////////////


        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]


        self.plot_confusion_matrix(y_test, y_pred, ['No Stroke', 'Stroke'], 'Random Forest')

        self.model_performance['Random Forest'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            # Include other metrics here as needed
        }
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        self.roc_data['Random Forest'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store metrics in the model_performance dictionary
        self.model_performance1['Random Forest'] = {
             'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC_AUC': roc_auc
        }
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Random Forest (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Random Forest')
        plt.legend(loc='lower right')
        plt.savefig("random_forest_roc_curve.png")
        plt.close()


        cv_scores_mean = cv_scores.mean()

        self.k_fold_means['Random Forest'] = cv_scores.mean()

        result = (
            f"Classification_rep:\n{classification_rep}\n"
        )
        result1= result
        print(f"This is for random forest, {result1}")
        return result


    def train_knn(self):
        global balanced_dfs_global

        if balanced_dfs_global.empty:
            print("Stratified dataset is empty. Please perform stratified sampling and cleaning first.\n")
            return

        selected_features = self.perform_chi_square_test()
        if not selected_features:  # If no features were selected
            print("No features selected by chi-square test.\n")
            return

        # Prepare the data
        X = balanced_dfs_global[selected_features]
        y = balanced_dfs_global['Stroke']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        unique1, counts1 = np.unique(y_train, return_counts=True)
        print(f"Class distribution before SMOTE: {dict(zip(unique1, counts1))}")
        # ///////////////////SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Check new class distribution
        unique, counts = np.unique(y_train_smote, return_counts=True)
        print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")

        # Initialize the KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=5)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')

        # Can use other scoring metrics as needed
        self.k_fold_results['KNN'] = cv_scores
        # Fit the model
        knn.fit(X_train_smote, y_train_smote)

        """

        #//////////Orinigial
        # Initialize the KNN Classifier
        #knn = KNeighborsClassifier(n_neighbors=5)

        #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(knn, X_train, y_train, cv=skf,
        #                            scoring='accuracy')  # Can use other scoring metrics as needed
        #self.k_fold_results['KNN'] = cv_scores
        # Fit the model
        #knn.fit(X_train, y_train)
        """
        #///////////////////ENN
        #enn = EditedNearestNeighbours()
        #X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

        # Check how the data has been balanced
        #unique, counts = np.unique(y_train_enn, return_counts=True)
        #print(f"Balanced class counts after ENN: {dict(zip(unique, counts))}")

        # Initialize the KNN Classifier
        #knn = KNeighborsClassifier(n_neighbors=3)

        ##skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #cv_scores = cross_val_score(knn, X_train_enn, y_train_enn, cv=skf, scoring='accuracy')
        # Can use other scoring metrics as needed
        #self.k_fold_results['KNN'] = cv_scores
        # Fit the model
        #knn.fit(X_train_enn, y_train_enn)

        #///////////////////////


        # Make predictions
        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)[:, 1]

        # Find optimal K


        self.plot_confusion_matrix(y_test, y_pred, ['No Stroke', 'Stroke'], 'KNN')

        self.model_performance['KNN'] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            # Include other metrics here as needed
        }
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Store metrics in the model_performance dictionary
        self.model_performance1['KNN'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC_AUC': roc_auc
        }
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        self.roc_data['KNN'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'KNN (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')

        plt.savefig("knn_roc_curve.png")
        plt.close()  # Close the plot to free up memory

        target_variable = 'Stroke'

        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plotting and saving the plots
        for feature in selected_features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=balanced_dfs_global[feature], y=balanced_dfs_global[target_variable], hue=balanced_dfs_global[target_variable], palette="viridis", alpha=0.6)
            plt.title(f'{feature} vs {target_variable}')
            plt.xlabel(f'{feature}')
            plt.ylabel(f'{target_variable}')

            # Save the plot
            plt.savefig(f"{feature}_vs_{target_variable}.png")

            # Close the plot
            plt.close()

        cv_scores_mean = cv_scores.mean()
        self.k_fold_means['KNN'] = cv_scores.mean()
        self.plot_model_performance()
        self.plot_performance_metrics()

        result = (
            f"Classification_rep:\n{classification_rep}\n"
        )
        result1 = result
        print(f"This is for knn, {result1}")
        return result


    def show_predictions(self):
        selected_option = self.optionmenu_1_page2.get()
        image_path = self.get_image_path(selected_option)
        self.display_image(image_path)


    def display_image(self, image_path):

        if not os.path.exists(image_path):
            print(f"Error loading image: {image_path} does not exist")
            messagebox.showerror("Error", f"Cannot load image: {image_path} does not exist")
            return
        try:
            image = Image.open(image_path)
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label_page2.configure(image=photo)
            self.image_label_page2.image = photo  # Reference to avoid garbage collection
        except Exception as e:
            print(f"Error loading image: {e}")

    def update_visualization_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label_page2.configure(image=photo)
            self.image_label_page2.image = photo
        except Exception as e:
            print(f"Error loading image: {e}")

    def plot_confusion_matrix(self, y_true, y_pred, classes, model_name):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if not hasattr(self, 'conf_matrices'):
            self.conf_matrices = {}
        self.conf_matrices[model_name] = cm

    def plot_all_confusion_matrices(self):
        if not hasattr(self, 'conf_matrices'):
            print("Confusion matrices not available.")
            return

        num_models = len(self.conf_matrices)
        # Determine the layout of subplots
        if num_models <= 2:
            fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 8))  # Increased figure size
        else:
            rows = 2
            cols = num_models // rows + (num_models % rows > 0)
            fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))  # Increased figure size

        if num_models == 1:
            axes = [axes]  # Make axes iterable if only one model

        axes = axes.flatten()  # Flatten the axes array for easy iteration
        for ax, (model_name, cm) in zip(axes, self.conf_matrices.items()):
            # Reorder confusion matrix
            cm = cm[[1, 0], :][:, [1, 0]]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        annot_kws={"size": 30})  # Adjusted annot_kws for fontsize of annotations
            ax.set_title(model_name, fontsize=30)  # Increased font size for titles
            ax.set_xlabel('Predicted', fontsize=30)  # Increased font size for x-label
            ax.set_ylabel('True', fontsize=30)  # Increased font size for y-label
            ax.set_xticklabels(['Stroke', 'No Stroke'], fontsize=30)  # Adjusted font size for x-tick labels
            ax.set_yticklabels(['Stroke', 'No Stroke'], fontsize=30)  # Adjusted font size for y-tick labels

        # Hide any extra subplots that aren't needed
        for i in range(num_models, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(pad=3.0)  # Added padding around subplots
        plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between subplots

        plt_path = "combined_confusion_matrix.png"
        plt.savefig(plt_path)
        plt.close()

        return plt_path

    def find_optimal_k(self, X_train, X_test, y_train, y_test, max_k=40):
        from sklearn.neighbors import KNeighborsClassifier
        error_rates = []

        for i in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rates.append(np.mean(pred_i != y_test))

        return error_rates

    def show_selected_figure(self):
        model_name = self.model_optionmenu.get()
        figure_type = self.figure_type_optionmenu.get()

        # Adjust the logic to display the overall ROC curve image
        if model_name == "Overall" and figure_type == "ROC Curve":
            image_path = "overall_roc_curve.png"
        elif model_name == "KNN" and figure_type == "KNN Error Rate":
            image_path = "knn_roc_curve.png"
        elif model_name == "KNN" and figure_type == "KNN Decision Boundaries":
            image_path = "overall_roc_curve.png"
        else:
            model_name_formatted = model_name.replace(" ", "_").lower()  # Standardize the model name format
            figure_type_formatted = figure_type.replace(" ", "_").lower()
            image_path = f"{model_name_formatted}_{figure_type_formatted}.png"

        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Cannot load image: {image_path} does not exist")
            return

        try:
            image = Image.open(image_path)
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label_page2.configure(image=photo)
            self.image_label_page2.image = photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {image_path}\n{e}")

    def update_available_figures(self, event=None):
        # Example model-to-figures mapping (assuming all models have the same figures for simplicity)
        model_figures = {
            "Overall": ["ROC Curve"],
            "Decision Tree": ["ROC Curve", "Confusion Matrix"],
            "KNN": ["ROC Curve", "Confusion Matrix","KNN Error Rate", "KNN Decision Boundaries"],
            "Logistic Regression": ["ROC Curve", "Confusion Matrix"],
            "Random Forest": ["ROC Curve", "Confusion Matrix"]
            # If models had different available figures, they would be listed differently above
        }

        selected_model = self.model_optionmenu.get()
        available_figures = model_figures[selected_model]

        # Update figure type optionmenu with the available figures for the selected model
        self.figure_type_optionmenu.set_values(available_figures)
        self.figure_type_optionmenu.set(available_figures[0])  # Set to first available figure as default

    def show_prediction(self):
        if not self.logisticregression_mode:
            self.prediction_label.configure(text="Please train the model first.")
            return

        feature_values = []
        for feature, menu in self.dynamic_option_menus.items():
            selected_label = menu.get()  # Get the selected label from UI
            # Find the corresponding value
            for value, label in self.feature_values[feature]:
                if label == selected_label:
                    feature_values.append(value)
                    break

        # Ensure the feature values are in the correct format for your model
        # This might involve converting them to floats or integers, reshaping, etc.
        features = np.array([feature_values]).astype(np.float32)  # Adjust type as needed

        # Predict using the decision tree model
        prediction = self.logisticregression_mode.predict(features)
        prediction_proba = self.logisticregression_mode.predict_proba(features)

        # Get the probability for the predicted class
        stroke_probability = prediction_proba[0][prediction[0]] * 100  # Convert to percentage

        # Display the result with probability
        result_text = f"Risk of Stroke: {'Yes' if prediction[0] == 1 else 'No'}\nProbability: {stroke_probability:.2f}%"
        self.prediction_label.configure(text=result_text)

    def plot_model_performance(self):


        # Dynamically determine available metrics to avoid KeyError
        metrics = set()
        for model_metrics in self.model_performance.values():
            metrics.update(model_metrics.keys())
        metrics = list(metrics)

        performances = {metric: [] for metric in metrics}
        models = list(self.model_performance.keys())

        # Collecting performances for each metric
        for model in models:
            for metric in metrics:
                performances[metric].append(self.model_performance[model].get(metric, np.nan))

        # Plotting
        x = np.arange(len(models))
        width = 0.15

        fig, ax = plt.subplots()
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, performances[metric], width, label=metric)

        ax.set_xlabel('Models')
        ax.set_title('Performance Metrics by Model')
        ax.set_xticks(x + width * len(metrics) / 2 - width / 2)
        ax.set_xticklabels(models)
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0))

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig("overall_roc_curve.png")
        plt.close()

    def create_dynamic_ui_elements(self):

        if not hasattr(self, 'selected_features') or not self.selected_features:
            # selected_features not set or empty; return early or display a message
            print("Selected features not yet determined.")
            return
        row = 2  # Starting row for dynamic elements
        self.dynamic_option_menus = {}  # Resetting dynamic option menus
        for feature, options in self.feature_values.items():
            if feature not in self.selected_features:  # Skip if feature is not selected
                continue
            label_text = f"This is for {feature}"
            label = customtkinter.CTkLabel(self.main_page_frame, text=label_text, font=("Helvetica", 14))
            label.grid(row=row, column=0, padx=20, pady=(5, 0), sticky="w")

            option_values = [label for value, label in options]  # Getting option labels
            option_menu = customtkinter.CTkOptionMenu(self.main_page_frame, values=option_values,
                                                      dynamic_resizing=False)
            option_menu.grid(row=row + 1, column=0, padx=20, pady=5, sticky="we")

            self.dynamic_option_menus[feature] = option_menu
            row += 2
        self.predict_button = customtkinter.CTkButton(self.main_page_frame, text="Predict Stroke",
                                                      command=self.show_prediction)
        self.predict_button.grid(row=row+1, column=0, pady=50, padx=20, sticky="we")

        self.prediction_label = customtkinter.CTkLabel(self.main_page_frame, text="", font=("Helvetica", 20),
                                                       wraplength=750,
                                                       justify=tkinter.LEFT)
        self.prediction_label.grid(row=row+1, column=1, columnspan=2, pady=(5, 0), padx=10, sticky="w")

    def show_frame(self, frame):
        frame.tkraise()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)



    def set_text(self, widget, text):
        """ Clear the widget and set new text. """
        widget.delete('1.0', tkinter.END)  # Clear existing text
        widget.insert(tkinter.END, text)  # Insert new text

    def load_clean_and_display(self):
        # Load data and display in label_1
        #load_result = self.load_and_display_data()
        #self.label_1.configure(text=load_result)  # Update text on label_1
        self.load_and_display_data()

        pie_chart_path = self.create_pie_chart()
        self.display_pie_chart(pie_chart_path)
        # Clean data and display in label_2
        self.clean_data()
        #clean_result = self.clean_data()
        #self.label_2.configure(text=clean_result)  # Update text on label_2

        self.impute_data()
        # Impute data
        #impute_result = self.impute_data()
        #self.label_3.configure(text=impute_result)

        # Perform stratified sampling
        self.perform_stratified_sampling()


        pie_chart_path2 = self.create_pie_chart2()
        self.display_pie_chart2(pie_chart_path2)

        """
        pie_chart_path3 = self.create_pie_chart2()
        self.display_pie_chart3(pie_chart_path3)
        """
        # Perform chi-square test
        chi_square_result = self.perform_chi_square_test()


        self.setup_chi_square_treeview(self.widget_5)
        self.plot_feature_distribution()
        self.plot_feature_vs_stroke()

    def train_all_models(self):

        """
        # Train Decision Tree and display the results
        decision_tree_result = self.train_decision_tree()
        self.label_11.configure(text=decision_tree_result)

        # Train Logistic Regression and display the results
        logistic_regression_result = self.train_logistic_regression()
        self.label_12.configure(text=logistic_regression_result)

        # Train Random Forest and display the results
        random_forest_result = self.train_random_forest()
        self.label_13.configure(text=random_forest_result)

        # Train KNN and display the results
        knn_result = self.train_knn()
        self.label_14.configure(text=knn_result)
        """
        self.train_decision_tree()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_knn()
        #self.display_k_fold_results()
        self.plot_k_fold_results()
        self.plot_roc_comparison()
        self.plot_all_confusion_matrices()
        self.display_specific_graphs()
        """self.combine_graphs_into_grids()"""
    def update_graph_display(self, selected_graph):
        if selected_graph == "K-Fold Results":
            self.plot_k_fold_results()  # Assuming this is already defined to plot K-Fold results
        elif selected_graph == "ROC Comparison":
            self.plot_roc_comparison()  # New method to be defined

    def display_graph_in_option_frame(self, selected_graph):
        # Mapping from the option menu to file names
        filename_map = {
            "Overall Confusion Matrix": "combined_confusion_matrix.png",
            "Decision Tree ROC Curve": "decision_tree_roc_curve.png",
            "KNN ROC Curve": "knn_roc_curve.png",
            "Logistic Regression ROC Curve": "logistic_regression_roc_curve.png",
            "Random Forest ROC Curve": "random_forest_roc_curve.png",
            "Overall Model Comparison": "overall_roc_curve.png",
            "Model Performance Metrics": "model_performance_metrics.png",
        }

        # Use the selected graph to get the correct filename
        image_path = filename_map.get(selected_graph, "File not found")

        # Display the image
        self.display_image_in_option_frame(image_path)

    def display_image_in_option_frame(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((600, 500), Image.Resampling.LANCZOS)  # Adjust size as needed for the option frame
            photo = ImageTk.PhotoImage(image)
            self.graph_label1.configure(image=photo)
            self.graph_label1.image = photo  # Keep a reference to avoid garbage collection
        except FileNotFoundError:
            self.graph_label1.configure(text=f"File not found: {image_path}")
            print(f"File not found: {image_path}")  # Debug print
        except Exception as e:
            self.graph_label1.configure(text=f"Error loading image: {e}")
            print(f"Error loading image: {e}")  # Debug print

    def display_image_in_label(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((500, 500), Image.Resampling.LANCZOS)  # Resize as needed
            photo = ImageTk.PhotoImage(image)
            self.graph_label.configure(image=photo)
            self.graph_label.image = photo  # Keep a reference
        except FileNotFoundError:
            self.graph_label.configure(text=f"File not found: {image_path}")
        except Exception as e:
            self.graph_label.configure(text=f"Error loading image: {e}")

    def display_k_fold_results(self):
        display_text = ""
        for model_name, scores in self.k_fold_results.items():
            display_text += f"{model_name} K-Fold Results:\n"
            for i, score in enumerate(scores, start=1):
                display_text += f"Fold {i}: {score:.4f}\n"
            mean_score = self.k_fold_means.get(model_name, 0)
            display_text += f"Mean K-Fold Score: {mean_score:.4f}\n\n"

        self.k_fold_results_label.configure(text=display_text)

    def plot_k_fold_results(self):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'red', 'purple']
        for i, (model, scores) in enumerate(self.k_fold_results.items()):
            x = np.arange(len(scores)) + 1  # Folds are 1-indexed for display
            ax.plot(x, scores, label=f'{model} (mean: {np.mean(scores):.2f})', marker='o', color=colors[i])

        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('K-Fold Cross-Validation Results')
        ax.legend(loc='center right')
        plt.xticks(x)

        plt.savefig('kfold.png', dpi=300)
        # Convert the Matplotlib figure to a Tkinter PhotoImage
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = Image.frombytes('RGB', (int(width), int(height)), canvas.tostring_rgb())
        photo = ImageTk.PhotoImage(image=img.resize((400, 400), Image.Resampling.LANCZOS))  # Resize the image

        """
        # Display the image in the designated Label widget
        self.graph_label.configure(image=photo)
        self.graph_label.image = photo  # Keep a reference to avoid garbage collection
         """
        plt.close(fig)  # Close the figure to free memory

    def plot_roc_comparison(self):
        # Create figure for plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['blue', 'green', 'red', 'purple']  # Assign different colors for each model

        for model_name, data in self.roc_data.items():
            ax.plot(data['fpr'], data['tpr'], label=f'{model_name} (ROC_AUC = {data["roc_auc"]:.2f})', color=colors.pop(0))

        ax.plot([0, 1], [0, 1], 'k--')  # Diagonal dashed line for no-skill classifier
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc='lower right')
        ax.grid(True)

        # Save the plot to a file
        plt.savefig('roc_comparison.png', dpi=300)
        # Convert the Matplotlib figure to a Tkinter PhotoImage
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = Image.frombytes('RGB', (int(width), int(height)), canvas.tostring_rgb())
        photo = ImageTk.PhotoImage(image=img.resize((400, 400), Image.Resampling.LANCZOS))

        """
        # Display the image in the designated Label widget
        self.graph_label.configure(image=photo)
        self.graph_label.image = photo  # Keep a reference to avoid garbage collection
         """
        plt.close(fig)  # Close the figure to free memory

    def setup_chi_square_treeview(self, parent):
        # Safety check to ensure chi_square_results are ready
        if not hasattr(self, 'chi_square_results') or not self.chi_square_results:
            print("Chi-square results are not ready. Please run the chi-square test first.")
            return
        customFont = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # Configure the style
        style = ttk.Style()
        style.theme_use("default")

        # Configure Treeview style with custom font
        style.configure("Custom.Treeview", font=customFont, background="#2a2d2e", foreground="white",
                        rowheight=25, fieldbackground="#343638")
        style.map('Custom.Treeview', background=[('selected', '#22559b')])

        # Configure Treeview Heading style with custom font
        style.configure("Custom.Treeview.Heading", font=customFont, background="#565b5e", foreground="white",
                        relief="flat")
        style.map("Custom.Treeview.Heading", background=[('active', '#3484F0')])

        # Create the Treeview widget
        tree = ttk.Treeview(parent, columns=("Variable", "Chi2", "p-value", "DoF", "Cramer's V", "Result"),
                            show="headings")
        for col in ["Variable", "Chi2", "p-value", "DoF", "Cramer's V", "Result"]:
            tree.heading(col, text=col)
            tree.column(col, width=125)

        sorted_data = sorted(self.chi_square_results.items(), key=lambda item: item[1][0], reverse=True)
        for variable, (chi2_stat, p, dof, cramers_v, result) in sorted_data:
            formatted_chi2_stat = f"{chi2_stat:.2f}"
            formatted_p = f"{p:.2f}"
            formatted_cramers_v = f"{cramers_v:.2f}"
            tree.insert("", "end",
                        values=(variable, formatted_chi2_stat, formatted_p, dof, formatted_cramers_v, result))

        tree.pack(expand=True, fill='both')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)

    import matplotlib.pyplot as plt

    def create_pie_chart(self):
        if 'Stroke' not in dfs.columns:
            print("Stroke column missing from the data")
            return

        stroke_counts = dfs['Stroke'].value_counts()
        total_strokes = stroke_counts.sum()  # Calculate total number of entries for stroke data
        labels = ['No Stroke', 'Stroke']  # Generic labels for the legend
        colors = ['#ff9999', '#66b3ff']  # Colors corresponding to each category

        # Define a custom function for autopct to display both percentage and count
        def make_autopct(counts):
            def my_autopct(pct):
                total = sum(counts)
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n({val})'

            return my_autopct

        fig1, ax1 = plt.subplots()
        wedges, texts, autotexts = ax1.pie(stroke_counts, colors=colors,
                                           autopct=make_autopct(stroke_counts),
                                           startangle=90)

        # Customize text properties for the pie chart percentages and counts
        plt.setp(autotexts, size=20, color='black')

        # Convert pixels to points, assuming you want 16px for the legend title
        px_to_pt = lambda px: px * 0.75
        legend_title_size_pt = px_to_pt(24)  # Convert 16px to points

        # Append total count to the legend title
        legend_title = f"Total: {total_strokes}"

        # Create a legend with larger font size for items and the title
        ax1.legend(wedges, labels, title=legend_title, loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=20, title_fontsize=legend_title_size_pt)

        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig1.patch.set_alpha(0.0)  # Set the figure background to transparent
        ax1.patch.set_alpha(0.0)  # Set the axis background to transparent

        plt.savefig("stroke_pie_chart.png", transparent=True,
                    bbox_inches='tight')  # Save with transparent background and adjust bounding box
        plt.close(fig1)  # Close the plot to free up memory
        return "stroke_pie_chart.png"  # Return the path of the saved image

    def create_pie_chart2(self):
        global balanced_dfs_global
        if 'Stroke' not in balanced_dfs_global.columns:
            print("Stroke column missing from the data")
            return

        stroke_counts = balanced_dfs_global['Stroke'].value_counts()
        total_strokes = stroke_counts.sum()  # Calculate total number of entries for stroke data
        labels = ['No Stroke', 'Stroke']  # Generic labels for the legend
        colors = ['#ff9999', '#66b3ff']  # Colors corresponding to each category

        # Define a custom function for autopct to display both percentage and count
        def make_autopct(counts):
            def my_autopct(pct):
                total = sum(counts)
                val = int(round(pct * total / 100.0))
                return f'{pct:.1f}%\n({val})'

            return my_autopct

        fig1, ax1 = plt.subplots()
        wedges, texts, autotexts = ax1.pie(stroke_counts, colors=colors,
                                           autopct=make_autopct(stroke_counts),
                                           startangle=90)

        # Customize text properties for the pie chart percentages and counts
        plt.setp(autotexts, size=20, color='black')

        # Convert pixels to points, assuming you want 16px for the legend title
        px_to_pt = lambda px: px * 0.75
        legend_title_size_pt = px_to_pt(24)  # Convert 16px to points

        # Append total count to the legend title
        legend_title = f"Total: {total_strokes}"

        # Create a legend with larger font size for items and the title
        ax1.legend(wedges, labels, title=legend_title, loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=20, title_fontsize=legend_title_size_pt)

        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig1.patch.set_alpha(0.0)  # Set the figure background to transparent
        ax1.patch.set_alpha(0.0)  # Set the axis background to transparent

        plt.savefig("stroke_pie_chart2.png", transparent=True,
                    bbox_inches='tight')  # Save with transparent background and adjust bounding box
        plt.close(fig1)  # Close the plot to free up memory
        return "stroke_pie_chart2.png"  # Return the path of the saved image



    def display_pie_chart(self, image_path):
        from PIL import Image, ImageTk
        img = Image.open(image_path)
        img = img.resize((325, 200), Image.Resampling.LANCZOS) # Resize as needed
        img_photo = ImageTk.PhotoImage(img)
        self.graph_pie.configure(image=img_photo)
        self.graph_pie.image = img_photo  # Keep a reference

    def display_pie_chart2(self, image_path):
        from PIL import Image, ImageTk
        img = Image.open(image_path)
        img = img.resize((325, 200), Image.Resampling.LANCZOS)  # Resize as needed
        img_photo = ImageTk.PhotoImage(img)
        self.graph_pie2.configure(image=img_photo)
        self.graph_pie2.image = img_photo  # Keep a reference

    def display_pie_chart3(self, image_path):
        from PIL import Image, ImageTk
        img = Image.open(image_path)
        img = img.resize((325, 250), Image.Resampling.LANCZOS)  # Resize as needed
        img_photo = ImageTk.PhotoImage(img)
        self.graph_pie3.configure(image=img_photo)
        self.graph_pie3.image = img_photo  # Keep a reference

    def plot_performance_metrics(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # Prepare data
        models = list(self.model_performance1.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC']
        data = {metric: [self.model_performance1[model][metric] for model in models] for metric in metrics}

        # Number of groups and bar width
        n_groups = len(models)
        fig, ax = plt.subplots(figsize=(20, 8))  # Increased figure size for better readability
        index = np.arange(n_groups)
        bar_width = 0.15  # Reduced bar width
        opacity = 0.8

        # Creating bars for each metric
        for i, metric in enumerate(metrics):
            bars = plt.bar(index + i * bar_width, data[metric], bar_width, alpha=opacity, label=metric)

            # Add text annotations on the bars with adjusted font size
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom',  # va: vertical alignment
                         ha='center', fontsize=18)  # ha: horizontal alignment, adjusted font size here

        # Setting font sizes
        plt.xlabel('Model', fontsize=18)  # Font size for the X-axis label
        plt.ylabel('Scores', fontsize=18)  # Font size for the Y-axis label
        plt.title('Performance Metrics by Model', fontsize=18)  # Font size for the title
        plt.xticks(index + bar_width * (len(metrics) - 1) / 2, models, fontsize=18)  # Font size for the x-tick labels
        plt.yticks(fontsize=18)  # Font size for the y-tick labels

        # Adjusting legend position and font size
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=18)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Adjusting bottom to give more space for x-tick labels

        # Optionally, save the plot if needed
        plt.savefig('model_performance_metrics.png')
        plt.close()

    def update_stacked_bar_chart(self, selected_variable):
        try:
            global dfs, balanced_dfs_global

            if selected_variable not in dfs.columns or selected_variable not in balanced_dfs_global.columns:
                print(f"Variable {selected_variable} not found in datasets.")
                return

            # Prepare data for plotting
            prop_before = pd.crosstab(dfs[selected_variable], dfs['Stroke'], normalize='index').sort_index(axis=1,
                                                                                                           ascending=False)
            prop_after = pd.crosstab(balanced_dfs_global[selected_variable], balanced_dfs_global['Stroke'],
                                     normalize='index').sort_index(axis=1, ascending=False)

            # Create a mapping for the current selected variable's value labels
            value_labels = {code: label for code, label in self.feature_values[selected_variable]}

            # Update index for both dataframes to use value labels
            prop_before.index = prop_before.index.map(value_labels)
            prop_after.index = prop_after.index.map(value_labels)

            # Define colors such that the 'Yes' Stroke comes first


            # Plotting
            fig, ax = plt.subplots()
            prop_before.plot(kind='bar', stacked=True, color=[ 'green','red'], ax=ax, position=1, width=0.4)
            prop_after.plot(kind='bar', stacked=True, color=[ 'lightgreen','lightcoral'], ax=ax, position=0, width=0.4)
            # Setting title, xlabel, and ylabel with font sizes
            plt.title(f'Comparison of Stroke Prevalence Before and After Stratified \nSampling for {selected_variable} Against Stroke', fontsize=14)
            plt.xlabel('Variable', fontsize=12)
            plt.ylabel('Proportion', fontsize=14)
            # Here is where you set the y-axis label size
            ax.tick_params(axis='y', labelsize=14)  # Set the y-axis label size to 18

            plt.legend(["Stroke - Yes (Before)", "Stroke - No (Before)", "Stroke - Yes (After)", "Stroke - No (After)"],
                       loc='upper right', bbox_to_anchor=(1, 1))

            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, horizontalalignment='right', fontsize=13)
            ax.set_xlim(left=-0.5)

            plt.tight_layout()

            # Save the plot
            plt_path = f"{selected_variable}_stacked_bar_chart.png"
            plt.savefig(plt_path)
            plt.close()

            # Display the plot
            self.display_image_in_DPlabel(plt_path, self.graph_labelDP)

        except Exception as e:
            print("Failed to update stacked bar chart:", e)

    def display_image_in_DPlabel(self, image_path, label_widget):
        from PIL import Image, ImageTk  # Ensure these are imported at the top of your file if not already
        try:

            # Load the image using PIL and resize it
            image = Image.open(image_path)
            image = image.resize((475, 475), Image.Resampling.LANCZOS)  # Resize as needed

            # Create a PhotoImage object, which is tkinter compatible
            photo = ImageTk.PhotoImage(image)


            # Configure the label widget to show the image
            label_widget.configure(image=photo)
            label_widget.image = photo  # Keep a reference to prevent garbage collection

        except FileNotFoundError:
            label_widget.configure(text=f"File not found: {image_path}")
        except Exception as e:
            label_widget.configure(text=f"Error loading image: {e}")



    def show_frame1(self, frame):
        frame.tkraise()
        self.load_clean_and_display()
        self.train_all_models()
        self.create_dynamic_ui_elements()
    def show_frame2(self, frame):
        frame.tkraise()


    def dynamic_samplingprop(self, selected_variable=None):
        if selected_variable:

            self.update_stacked_bar_chart(selected_variable)

    def display_specific_graphs(self):
        """
        Display predefined images in specific customtkinter widgets.
        """
        # Define the mapping of widgets to images
        images = {
            'graph_option_frame2': 'kfold.png',
            'graph_display_frame2': 'model_performance_metrics.png',
            'graph_option_frame3': 'combined_confusion_matrix.png',
            'graph_display_frame3': 'roc_comparison.png'

        }

        # Iterate over the mapping to display each image in the corresponding widget
        for widget_name, image_path in images.items():
            # Ensure the image file exists
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                continue

            # Load the image
            image = Image.open(image_path)
            image = image.resize((475, 350), Image.Resampling.LANCZOS)  # Resize as needed
            photo = ImageTk.PhotoImage(image)

            # Get the widget based on its name
            widget = getattr(self, widget_name)

            # Create a CTkLabel if not already exists or update existing one
            if hasattr(widget, 'image_label'):
                widget.image_label.configure(image=photo)
            else:
                widget.image_label = customtkinter.CTkLabel(widget, image=photo)
                widget.image_label.pack(fill="both", expand=True)

            # Keep a reference to the image to avoid garbage collection
            widget.image_label.image = photo

    def plot_feature_distribution(self):
        global balanced_dfs_global
        if balanced_dfs_global.empty:
            print("Balanced dataset is empty. Please check data loading and preprocessing steps.")
            return

        # Ensure at least two features are selected
        if len(self.selected_features) < 2:
            print("Not enough features selected to plot interactions. Select at least two features.")
            return

        # Select the first two features for simplicity
        feature1, feature2 = self.selected_features[0], self.selected_features[1]
        pivot_data = pd.pivot_table(balanced_dfs_global, values='Stroke', index=[feature1, feature2],
                                    aggfunc=lambda x: np.mean(x))

        # Reset index to convert multi-index into columns
        pivot_data.reset_index(inplace=True)
        pivot_data['Stroke'] *= 100  # Convert probability to percentage

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature1, y='Stroke', hue=feature2, data=pivot_data)
        plt.title(f'Stroke Occurrence Percentage by {feature1} and {feature2}')
        plt.ylabel('Percentage of Stroke Occurrences')
        plt.xlabel(feature1)
        plt.legend(title=feature2)
        plt.xticks(rotation=45)  # Rotate labels for better readability

        plt_path = f"interaction_bar_{feature1}_{feature2}.png"
        plt.savefig(plt_path)
        plt.close()

    def plot_feature_vs_stroke(self):
        global balanced_dfs_global
        if balanced_dfs_global.empty:
            print("Balanced dataset is empty. Please check data loading and preprocessing steps.")
            return

        # Ensure at least one feature is selected
        if len(self.selected_features) == 0:
            print("No features selected. Please select at least one feature to plot.")
            return

        for feature in self.selected_features:
            if feature in balanced_dfs_global.columns:
                # Create a count plot for each feature against Stroke occurrence
                plt.figure(figsize=(10, 6))
                ax = sns.countplot(x=feature, hue='Stroke', data=balanced_dfs_global)
                plt.title(f'Count of {feature} by Stroke Occurrence')
                plt.ylabel('Count')
                plt.xlabel(feature)
                plt.legend(title='Stroke', loc='upper right')

                # Annotate each bar with the count value
                for p in ax.patches:
                    height = p.get_height()
                    if pd.notna(height):  # Check for NaNs and place text accordingly
                        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

                # Use the feature_values dictionary to replace x-tick labels with meaningful names
                # Use the feature_values dictionary to replace x-tick labels with meaningful names
                label_dict = {int(k): v for k, v in
                              self.feature_values[feature]}  # Convert keys to int if they are not already
                ax.set_xticklabels(
                    [label_dict.get(int(float(t.get_text())), t.get_text()) for t in ax.get_xticklabels()])

                # Save the plot
                plt_path = f"{feature}_vs_stroke_distri.png"
                plt.savefig(plt_path)
                plt.close()
                print(f"Plot saved as {plt_path}")
            else:
                print(f"Feature {feature} is not in the dataset.")

    """
    def combine_graphs_into_grids(image_names, start_index, save_as):

        # Size of each individual image assuming uniform dimensions for simplicity
        width, height = 400, 300  # Adjust size based on your specific graph dimensions

        # Create a new blank image for the grid
        grid_image = Image.new('RGB', (width * 3, height * 3))

        # Loop through each position in a 3x3 grid
        for i in range(3):  # For each row
            for j in range(3):  # For each column
                # Calculate index in the image list
                index = start_index + i * 3 + j
                if index < len(image_names):
                    # Open the image
                    img = Image.open(image_names[index])
                    # Resize image if necessary
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    # Paste the image into the grid
                    grid_image.paste(img, (j * width, i * height))

        # Save the combined image
        grid_image.save(save_as)

    # Example usage
    # List of graph image filenames
    graph_images = [
        "Age_Group_stacked_bar_chart.png", "Current_ESmoker_stacked_bar_chart.png", "Depression_stacked_bar_chart.png",
        "Diabetes_stacked_bar_chart.png", "Fruit_Intake_stacked_bar_chart.png", "Gender_stacked_bar_chart.png",
        "Heart_Disease_stacked_bar_chart.png", "High_Blood_Pressure_stacked_bar_chart.png", "High_Cholesterol_stacked_bar_chart.png",
        "Income_Group_stacked_bar_chart.png", "Marital_Status_stacked_bar_chart.png", "PhysicalActivity_stacked_bar_chart.png",
        "Race_stacked_bar_chart.png", "Smoking_Status_stacked_bar_chart.png", "Urban_Status_stacked_bar_chart.png",
        "Vegetable_Intake_stacked_bar_chart.png", "BMI_Category_stacked_bar_chart.png", "Education_Level_stacked_bar_chart.png"
    ]

    # Create the first combined image of the first 9 graphs
    combine_graphs_into_grids(graph_images, 0, "combined_graphs_1.png")

    # Create the second combined image of the next 9 graphs
    combine_graphs_into_grids(graph_images, 9, "combined_graphs_2.png")
    """
if __name__ == "__main__":
    app = App()
    app.mainloop()