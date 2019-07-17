library(R6)
library(caret)
library(readr)
library(janitor)
library(dplyr)
library(parallel)
library(doParallel)

BaseModel <- 
	R6Class("BaseModel", 
		private = list(
		 	Name = NA,
			Type = NA
		),
		public = list(
			initialize = function(Name, Type) {
				if (!missing(Name)) { private$Name = Name }
				if (!missing(Type)) { private$Type = Type }
			},
			setName = function(Name) {
				if (!missing(Name)) { private$Name = Name }
			},
			setType = function(Type) {
				if (!missing(Type)) { private$Type = Type }
			},
				getName = function() {
				return(private$Name)
			},
			getType = function() {
				return(private$Type)
			 	}
			)
)

MLModel <- 
	R6Class("MLModel", inherit = BaseModel,
	   	public = list(
		Data = NULL,
		Method = NULL,
		TrainControl = NULL,
		TrainFit = NULL,
		SplitRate = NULL,
		TrainData = NULL,
		TestData = NULL,
		Target = NULL,
		TransTrain = NULL,
		TransTest = NULL,
		PreProcess = NULL,  
		TrainControlFlag = NULL,
		TurnGrid = NULL,
		Metric = NULL, 
		Predict = NULL,
		Performance = NULL,
		initialize = function(Name, Type) {
			super$initialize(Name, Type)
		},
		LoadDataWithSplitRate = function(Location, SplitRate, Target, Col2Remove) {
			self$Data <- readr::read_csv(Location)
			self$Data <- janitor::clean_names(self$Data)
			self$Data <- self$Data[,-Col2Remove]
			self$Target <- Target
			self$SplitRate <- SplitRate
			set.seed(12345)
			#folds <- split(sample(nrow(self$Data), nrow(self$Data), replace=TRUE), as.factor(1:3))
			#self$Data <- self$Data[folds[[1]],]
            self$Data <- sample_n(self$Data, nrow(self$Data) * 0.10, replace = F, prob = NULL )
            #self$Data <- self$Data[sample(nrow(self$Data) * 0.02, replace = FALSE, prob = NULL), ]
            #self$Data <- self$Data %>% ungroup() %>% sample_n(nrow(self$Data) * 0.10, replace = TRUE)
			TrainIndex <- createDataPartition(y=self$Data[[self$Target]], p=self$SplitRate, list = FALSE)
			self$TrainData <- self$Data[TrainIndex,]
			self$TestData <- self$Data[-TrainIndex,]
		},
		LoadDataWithDF = function(df){
			self$Data <- df
		},
		LoadDataFile = function(filename) {
			self$Data <- readr::read_csv(filename)
			self$Data <- janitor::clean_names(self$Data)
		},
		SplitData = function(SplitRate, Target) {
			self$Target <- Target
			self$SplitRate <- SplitRate
			self$Data <- janitor::clean_names(self$Data)
			#TrainIndex <- createDataPartition(y=self$Data[[Target]], p=SplitRate, list = FALSE)
			TrainIndex <- sample(nrow(self$Data), nrow(self$Data) * SplitRate/100, replace = FALSE, prob = NULL)
			self$TrainData <- self$Data[TrainIndex,]
			self$TestData <- self$Data[-TrainIndex,]
		},
		#RePlace = function(colname, from, to) {
		#	self$Data[self$Data$colname == from, ]$colname <- to  
		#}, 
		setMethod = function(Method) {
			self$Method <- Method
		},
		getMethod = function() {
			return(self$Method)
		},
		setTrainControl = function(TrainMethod, NumOfFold, Repeats, verboseIter){
		self$TrainControl <- trainControl(
			method = TrainMethod, 
			number = NumOfFold,
			repeats = Repeats,
			verboseIter = verboseIter
		)
		},	
		setTuneGrid = function(GridName, Metric) {
			self$TurnGrid = NULL
			if(GridName == "RF") { 
				self$TurnGrid = expand.grid(.mtry = c(1:ncol(self$Data)), .ntree = c(500, 1000, 1500, 2000, 2500) )
			} else if (GridName == "NB") {
				self$TurnGrid = expand.grid(.fL=c(seq(0.0,1.0,0.1)), .usekernel=c(TRUE,FALSE))
			}# else if (GridName == "SVM") {
			#	self$TurnGrid == expand.grid(.C=c(seq(0.001, 1, 100)))
			#} #else if (GridName == "KNN") {
			  #	self$TurnGrid == expand.grid(.k=c(1:sqrt(ncol(self$Data))))
			#}
			self$Metric <- Metric 
		},
		setPreProcess = function(PreMethod, verbose=TRUE){
			targetColNum = grep(self$Target, colnames(self$TrainData))
			if (is.null(PreMethod)) {
				self$TransTrain <- self$TrainData
				self$TransTest <- self$TestData
			} else {
				self$PreProcess <- preProcess(self$TrainData[,-c(targetColNum)], PreMethod )
				self$TransTrain <- predict(self$PreProcess, self$TrainData[,-c(targetColNum)])
				self$TransTest  <- predict(self$PreProcess, self$TestData[,-c(targetColNum)])
				self$TransTrain <- cbind(self$TransTrain, self$TrainData[,c(targetColNum)])
				self$TransTest  <- cbind(self$TransTest,  self$TestData[,c(targetColNum)])
			}
		},
		Regress = function(Features, TrainControlFlag=TRUE){
			print(self$Target)
			print(Features)
			if (typeof(Features) == "character") {
				formu <- as.formula(paste(self$Target, " ~ ", Features))
			} else {
				formu <- as.formula(paste(self$Target, " ~ ", paste(Features, collapse = "+")))
			}
			print(formu)
			print(self$Method)
			#self$TrainControl <- ifelse(TrainControlON, self$TrainControl, 'NONE')
			
			#cl <- makeCluster(detectCores()-2)
			#registerDoParallel(cl)
			
			if (TrainControlFlag) {
				self$TrainFit <- train(
				formu,
				method = self$Method, 
				trControl = self$TrainControl,
				data = self$TransTrain,
				allowParallel = TRUE,
				metric = self$Metric,
				tuneGrid = self$TuneGrid,
				na.action=na.omit
				)
			} else {
			self$TrainFit <- train(
				formu,
				method = self$Method, 
				#trControl = self$TrainControl,
				data = self$TransTrain,
				allowParallel = TRUE,
				metric = self$Metric,
				tuneGrid = self$TuneGrid,
				na.action=na.omit
				)
			}
			#summary(self$TrainFit)
			self$Predict <- predict(self$TrainFit, self$TransTest)
			if(identical(self$Matric, "Accuracy")) {
				self$Preformance <- confusionMatrix(data = self$Test[,self$Target], reference = self$Predict)
			} else if(identical(self$Matric,"RMSE")) {
				self$Preformance <- postResample(pred = self$Predict, obs = self$Test[,self$Target])
			} 	
			#stopCluster(cl)
			return(c(self$TrainFit, self$Performance))
			}
		)
)

