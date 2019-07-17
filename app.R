#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
source("./MLClass.R")

library(shiny)
library(reshape2)
library(klaR)
library(e1071)
library(kknn)
library(randomForest)

Regress <- function() {
	#observe({
	#	x <- c("60:40", "70:30", "80:20", "90:10", "95:5")
	#	y <- c(60, 70, 80, 90, 95)
	#	splitoption <- list()
	#	splitoption[x] <- y
	#})
	sidebarLayout(
		sidebarPanel(
			fluidRow(
				column(6,radioButtons("model", "Model", 
					choices = c("class_naive"="class_naive", "class_randomforest"="class_randomforest", "class_KNN"="class_KNN", "class_SVM"="class_SVM",
								"regress_randomforest"="regress_randomforest", "regress_KNN"="regress_KNN", "regress_SVM"="regress_SVM"), selected = c("") )),
									  	
				column(6,radioButtons("splitratio", "Split Ratio", choices = 
					c("60:40" = "60", "70:30" = "70", "80:20" = "80", "90:10" = "90", "95:5" = "95"), selected = "80"),
					textInput("seed", "Seed Value", value = "12345")
					)
			),
			fluidRow(
				column(6, uiOutput("parambutton")),
				column(6, uiOutput("paramcheck"))
			),
			width = 6
		),
		mainPanel(
			fluidRow(
				column(6, textOutput("selectedModel")),	
				column(6, textOutput("selectedFile"))
			), 
			fluidRow(
				column(6,textOutput("selectedTarget")),
				column(6,textOutput("selectedFeatures"))
			),
			fluidRow(
				column(6,textOutput("selectedSplitratio")),
				column(6,textOutput("selectedSeed"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput0"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput1"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput2"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput3"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput4"))
			),
			fluidRow(
				column(12, verbatimTextOutput("modelOutput5"))
			),
			#fluidRow(
			#	column(12, verbatimTextOutput("modelOutput6"))
			#),
			#fluidRow(
			#	column(12, verbatimTextOutput("modelOutput7"))
			#),
			
			
			width = 6
			#textOutput("textout1"),
			#uiOutput("uiout1")
		)
	)
}

RegressOut <- function(input, output, session) {
	#output$modelOutput1 <- renderText("Raj")
	#output$textout1 <- renderText("Raj")
	#output$uiout1 <- renderUI(input$xaxis)
	output$parambutton <- renderUI({
		radioButtons("target", "Target", choices = coloption, selected = input$xaxis)
	})
	output$paramcheck <- renderUI({
		checkboxGroupInput("features", "Features", choices = coloption, selected = input$yaxis)
	})
	
	output$selectedTarget <- renderText({ paste("Target :", input$target) })
	output$selectedFeatures <- renderText({ paste("Features :", paste(input$features, collapse = ", "))})
	output$selectedSplitratio <- renderText({ paste("Split Ratio :", input$splitratio)})
	output$selectedSeed <- renderText({ paste( "Seed :", input$seed) })
	output$selectedModel <- renderText({ paste( "Model :", input$model) })
	output$selectedFile <- renderText({ paste("File : ",  input$csvfile) })
}

Summary <- function() {
	sidebarLayout(
		sidebarPanel(
			fluidRow(
				column(6,radioButtons("xaxis", "X-Axis:", choices = c(None=""))),
				column(6,checkboxGroupInput("yaxis", "Y-Axis:", choices = c(None="")))
			),
			width = 6
		),
		mainPanel(
			tabsetPanel(
				tabPanel("Plot",plotOutput("plot")),
				tabPanel("Data", uiOutput("text101"), dataTableOutput("csvsummary"))
			), width = 6
		)
		
	)
}

SummaryOut <- function(input, output, session) {
	observe({
		csvcolnames <- names(csvdf())
		coloption <<- list()
		coloption[csvcolnames] <<- csvcolnames
		updateRadioButtons(session, "xaxis", "X-Axis:", choices = coloption, selected = "")
		updateCheckboxGroupInput(session, "yaxis", "Y-Axis", choices = coloption, selected = "")
	})
	
	output$csvsummary <- renderDataTable(
		csvdf()[,c(input$xaxis, input$yaxis)]
	)
	output$plot <- renderPlot({
		csvdf <- csvdf()
		gg <- NULL
		if (!is.null(csvdf())){
			xx <- input$xaxis
			yy <- input$yaxis
			if (!is.null(xx) & !is.null(yy)) {
				dat <- melt(csvdf(), id.vars=xx, measure.vars=yy)
				gg <- ggplot(data=dat) +
					geom_point(aes_string(x=xx, y="value", colour="variable"))
			}
		}
		return(gg)
	})
}

Dataset <- function(){
	sidebarLayout(
		sidebarPanel(
			fileInput("csvfile", "Choose CSV file", multiple = FALSE, 
					  accept = c(".csv", "text/csv")),
			radioButtons("quote", "Quote", 
						 choices = c(None="","Double"='"',"Single"="'"),
						 selected = '"'),
			radioButtons("display", "Display", 
						 choices = c(Head = "head", Tail="tail", All = "all"),
						 selected = "head"),
			uiOutput("columntoview")
			
		), 
		mainPanel(
			dataTableOutput("csvcontents")
		)
	)
}


DatasetOut <- function(input, output, session) {
	csvdf <<- reactive({ 
		req(input$csvfile)
		csvdf <- readr::read_csv(input$csvfile$datapath, quote =input$quote)
		csvdf <- janitor::clean_names(csvdf) 
	})
	
	output$columntoview <- renderUI({
		csvcolnames <<- names(csvdf())
		coloption <<- list()
		coloption[csvcolnames] <<- csvcolnames
		checkboxGroupInput("columntoview", "Columns to View", choices = coloption, selected = coloption)
	})
	
	output$csvcontents <- renderDataTable({
		if (input$display == "head") {
			return(head(csvdf()[,c(input$columntoview)]))
		} else if (input$display == "tail") {
			return(tail(csvdf()[,c(input$columntoview)]))
		} else { return(csvdf()[,c(input$columntoview)]) }
	})
}


# Define UI for application that draws a histogram
# Define UI for application 
ui <- navbarPage("My Data App",
				 navbarMenu("Dataset",
				 		   tabPanel("New Dataset", 
				 		   		 Dataset()		 		 
				 		   )
				 ),
				 navbarMenu("EDA", 
				 		   tabPanel("Summary",
				 		   		 Summary()
				 		   )
				 ),
				 navbarMenu("Regression/Classification",
				 		   tabPanel("Regress/Classfication", Regress())	   
				 		   #tabPanel("Naive Baise"),  #, Naive()),
				 		   #tabPanel("Random Forest"),
				 		   #tabPanel("KNN")
				 ),
				 #navbarMenu("Regression",
				 #		   #tabPanel("Parameter Selection", Regression()),	   
				 #		   tabPanel("Linear Regression")		   
				 #),
				 inverse = TRUE,
				 collapsible = TRUE
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
	csvdf <- reactive({ readr::read_csv(input$csvfile$datapath, quote =input$quote, na.rm=TRUE) })
	csvdf <- reactive({ janitor::clean_names(csvdf()) })
	DatasetOut(input, output,session)
	SummaryOut(input,output, session)
	RegressOut(input,output, session)
	#output$modelOutput1 <- renderText({ input$csvfile$datapath })
	perf <- reactive({
		filename <- input$csvfile$datapath
		splitrate <- as.numeric(input$splitratio) 
		target <- input$target
		features <-  as.list(paste(input$features, collapse = "+") )  
		col2remove <- c(1)
		ppmethods <- c("center", "scale")
		gridname = ""
		metric <- ifelse(startsWith(input$model, "class_"), "Accuracy", "RMSE")
		if(endsWith(input$model, "naive")) { method = "nb" }
		if(endsWith(input$model, "randomforest")) { method = "rf" }
		if(endsWith(input$model, "KNN")) { method = "kknn" }
		if(endsWith(input$model, "SVM")) { method = "svmLinear2" }
		
		model <- MLModel$new(method, metric)
		model$setTrainControl("repeatedcv", 5, 3, TRUE)
		model$setMethod(method)
		model$LoadDataFile(filename)
		#model$Data[model$Data$member_age_bucket=="32-25",]$member_age_bucket <- "32-35"
		#model$Data$amount_spent_per_day <- as.integer((model$Data$amount_spent_per_day + 50)/100) * 100
		model$SplitData(splitrate, target)
		#model$LoadDataWithSplitRate(filename, splitrate, target, col2remove)
		#df <- as.data.frame(csvdf())
		#model$LoadDataWithDF(df)
		model$setPreProcess(ppmethods)
		model$setTuneGrid(gridname, metric)
		
	
		perf <- model$Regress(features)	
	})
	
	output$selectedTarget <- renderText({ paste("Target :", input$target) })
	output$selectedFeatures <- renderText({ paste("Features :", paste(input$features, collapse = ", "))})
	output$selectedSplitratio <- renderText({ paste("Split Ratio :", input$splitratio)})
	output$selectedSeed <- renderText({ paste( "Seed :", input$seed) })
	output$selectedModel <- renderText({ paste( "Model :", input$model) })
	output$selectedFile <- renderText({ paste("File : ",  input$csvfile) })

	output$modelOutput0 <- renderPrint({ perf()$finalModel })
	output$modelOutput1 <- renderPrint({ perf()$results })
	output$modelOutput2 <- renderPrint({ perf()$modelType })
	output$modelOutput3 <- renderPrint({ perf()$resample })
	#output$modelOutput4 <- renderPrint({ perf()$modelInfo$fit })
	#output$modelOutput5 <- renderPrint({ perf()$modelInfo$prob })
	output$modelOutput4 <- renderPrint({ perf()$modelInfo$tags })
	output$modelOutput5 <- renderPrint({ perf()$bestTune  })
}

# Run the application 
shinyApp(ui = ui, server = server)

