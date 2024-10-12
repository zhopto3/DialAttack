#Data analysis for Word/Character Error Rates 
require(tidyverse)

#Read WER / CER data for each model into asr_d
asr_d <-data.frame()
models <- c("20", "50", "80", "100")

for (i in 1:length(models)) {
  di <- read_delim(paste0("asr/asr_", models[i], ".tsv"), delim = "\t", escape_double = FALSE, trim_ws = TRUE)
  #Add "model" factor to asr_d
  di$model <- models[i]
  asr_d <- rbind(asr_d, di)
}

#Replace WER/CER values of 0 with 0.00001:
asr_d<-asr_d %>%
  mutate(WER=ifelse(WER==0, 0.00001, WER))

asr_d<-asr_d %>%
  mutate(CER=ifelse(CER==0, 0.00001, CER))

###MODEL 1: sum-coded gamma model for WER###
asr_d$model <- factor(asr_d$model, levels = c("20", "50", "80", "100"))
contrasts(asr_d$model) <- contr.sum(4)
asr_d$Dialect <- factor(asr_d$Dialect, levels = c("balear","nord", "nord-occidental", "valencià", "central"))
contrasts(asr_d$Dialect) <- contr.sum(5)

#Contrasts leave out central and the 100% Central Catalan model
contrasts(asr_d$Dialect)
contrasts(asr_d$model)

#Fitting the gamma model using the identity link function
fit1<-glm(WER ~ model*Dialect, asr_d, family = Gamma(link="identity"))
summary(fit1)

###MODEL 2: sum-coded gamma model for CER###
fit2<-glm(CER ~ model*Dialect, asr_d, family = Gamma(link="identity"))
summary(fit2)

#Treatment contrasts for ASR: Compare everything to the Central asr in the 100 condition
contrasts(asr_d$model) <- contr.treatment(4, base=4)
contrasts(asr_d$model)

contrasts(asr_d$Dialect) <- contr.treatment(5, base = 5)
contrasts(asr_d$Dialect)

###MODEL 3: treatment-coded gamma model for WER###
fit3 <- glm(WER ~ model*Dialect, asr_d, family = Gamma(link="identity"))
summary(fit3)
# Interpreting the intercept coefficients is strange in these models, but you can restore the mean of each condition in fit3
# Example: the model1:dialect4 (20%, Valencia) coefficient is -0.121. To arrive at the mean for Valencia in the 20% model (0.125),
# do the following:
# intercept + model_1 + dialect_4 + model_1:dialect_4 = 0.152 + 0.012 + 0.082 + (-0.121) = 0.125
# If I understand right, this means we can't directly interpret the sign of the intercept coefficients. 

###MODEL 4: treatment-coded gamma model for CER###
fit4 <-glm(CER ~ model*Dialect, asr_d, family = Gamma(link="identity"))
summary(fit4)


#Data analysis for attack success rate
d <- data.frame()
models <- c("20", "50", "80", "100")

#Read attack success data into dataframe d
for (i in 1:length(models)) {
  di <- read_delim(paste0("attacks/attack_stats_", models[i], ".tsv"), delim = "\t", escape_double = FALSE, trim_ws = TRUE)
  #Add model factor to d
  di$model <- models[i]
  d <- rbind(d, di)
}

d_orig <- d
d$model <- factor(d$model, levels = c("20","50", "80", "100"))

#Sum coding, with Valencià and 100% model left out
contrasts(d$model) <- contr.sum(4)
d$dialect <- factor(d$dialect, levels = c("balear", "central", "nord","nord-occidental", "valencià"))
contrasts(d$dialect) <- contr.sum(5)

contrasts(d$dialect)

###MODEL 5: sum-coded binomial model for Attack success###
fit5 <- glm(success ~ model*dialect, d, family = "binomial")

summary(fit5)
#In fit5, we see the only significant coefficient is model3:dialect2, Central Catalan in the 80% model

#Sum coding for attack success again, but this time rotating the dialects and leaving out nord instead

d$dialect <- factor(d$dialect, levels = c("balear", "central","nord-occidental", "valencià", "nord"))
contrasts(d$dialect) <- contr.sum(5)

contrasts(d$dialect)
###MODEL 6: sum-coded binomial model for attack success (rotated)###
fit6 <- glm(success ~ model*dialect, d, family = "binomial")

summary(fit6)
# In this rotated model, we see the same significant coefficient; 

# Finally, we try treatment coding; Since we see that Central in the 80% model seems to be a bit harder to attack, 
# we directly compare everything to this condition
contrasts(d$model) <- contr.treatment(4, base=3)
contrasts(d$model)

contrasts(d$dialect) <- contr.treatment(5,base=2)
contrasts(d$dialect)
###MODEL 7: treatment-coded binomial model for attack success (compare to 80% Central, Central Dialect)###
fit7 <- glm(success ~ model*dialect, d, family = "binomial")
summary(fit7) 
# Interpretation of fit7;
# Can get the mean for the Central, 80% condition: 1/(1+exp(-intercept)) = 1/(1+exp(-1.1527))
# Ignoring marginal significance, we see three significant effects: model 1 (20%), model1:dialect3(20%, nord-oc), model1:dialect4 (20%, nord)
# So here we see that when we directly compare everything to the attack success rate with Central Catalan attacks on 80% Central model, there are not many conditions that are actually
# significantly easier to attack
