---
  title: "Assigment - Naive Bayes DIY"
author:
  - name author here - Yunus Gunes
- name reviewer here - Sjoerd Hamaker 
date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  html_notebook:
  toc: true
toc_depth: 2
---
  ## Business Understanding
For this business case we get a large dataset with several articles. These articles are either reliable or unreliable. We have build a model that analyses these articles and then identifies whether they are reliable or unreliable. 
## load packages
```{r}
# install.packages("tm")
# install.packages("caret")
# install.packages("wordcloud")
# install.packages("e1071")
```
## Data understanding
```{r}
library(tidyverse)
url <- "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/assignments/master/datasets/NB-fakenews.csv"
rawdf <- read_csv(url)
head(rawdf)
rawdf$label <- rawdf$label %>% factor %>% relevel("1")
class(rawdf$label)
table(rawdf$label)

s <- sample(c(1:dim(rawdf)[1], 5000))
reliable <- rawdf[s,] %>% filter(label == "0")
unreliable <- rawdf[s,] %>% filter(label == "1")
wordcloud::wordcloud(unreliable$text, max.words = 20, scale = c(4, 0.8), colors= c("indianred1","indianred2","indianred3","indianred"))
wordcloud::wordcloud(reliable$text, max.words = 20, scale = c(4, 0.8), colors= c("lightsteelblue1","lightsteelblue2","lightsteelblue3","lightsteelblue"))
```


## Data Preparation
```{r}
library(tm)
rawcorpus <- Corpus(VectorSource(rawdf$text))
inspect(rawcorpus[1:3])
```

```{r}
cleancorpus <- rawcorpus %>% tm_map(tolower) %>% tm_map(removeNumbers)

cleancorpus <- cleancorpus %>% tm_map(tolower) %>% tm_map(removeWords, stopwords()) %>% tm_map(removePunctuation)

cleancorpus <- cleancorpus %>% tm_map (stripWhitespace)
cleanDTM <- cleancorpus %>% DocumentTermMatrix()
inspect(cleanDTM)

freqWords <- cleanDTM %>% findFreqTerms(1000)
leanDTM <- DocumentTermMatrix(cleancorpus, list(dictionary = freqWords))
inspect(leanDTM)
```
## create split indices
```{r}
library(caret)
set.seed(1234)
trainIndex <- createDataPartition(rawdf$label, p=.75, list = FALSE, times = 1)
head(trainIndex)

trainDF <- rawdf[trainIndex, ]

testDF <- rawdf[-trainIndex, ]

trainCorpus <- cleancorpus[trainIndex]
testCorpus <- cleancorpus[-trainIndex]
trainDTM <- leanDTM[trainIndex, ]
testDTM <- leanDTM[-trainIndex, ]

# freqWords <- trainDTM %>% findFreqTerms(500)
# trainDTM <- DocumentTermMatrix(trainCorpus, list(dictionary = freqWords))

convert_count <- function(x) {x <- ifelse(x>0, 1, 0) %>% factor(levels = c(0, 1), labels = c("no", "yes"))}

nColsDTM <- dim(trainDTM)[2]
trainDTM <- apply(trainDTM, MARGIN =  2, convert_count)
testDTM <- apply(testDTM, MARGIN =  2, convert_count)
head(trainDTM[, 1:10])
```

## Modeling and Evaluation
```{r}
library(e1071)
nbayesModel <- naiveBayes(trainDTM, trainDF$label, laplace = 1)

predVec <- predict(nbayesModel, testDTM)

confusionMatrix(predVec, testDF$label, positive = "1", dnn = c("prediction", "true"))
```


