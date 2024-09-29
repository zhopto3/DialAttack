
require(tidyverse)
d <- data.frame()
models <- c("20", "50", "80", "100")

for (i in 1:length(models)) {
  di <- read_delim(paste0("attacks/attack_stats_", models[i], ".tsv"), delim = "\t", escape_double = FALSE, trim_ws = TRUE)
  di$model <- models[i]
  d <- rbind(d, di)
}

d_orig <- d
d$model <- factor(d$model, levels = c("20", "50", "80", "100"))
contrasts(d$model) <- contr.sum(4)
d$dialect <- factor(d$dialect, levels = c("balear", "nord", "nord-occidental", "valencià", "central"))
contrasts(d$dialect) <- contr.sum(5)

contrasts(d$dialect)

fit1 <- glm(success ~ model*dialect, d, family = "binomial")

summary(fit1)

#Read in the asr data 

asr_d <-data.frame()
models <- c("20", "50", "80", "100")

for (i in 1:length(models)) {
  di <- read_delim(paste0("asr/asr_", models[i], ".tsv"), delim = "\t", escape_double = FALSE, trim_ws = TRUE)
  di$model <- models[i]
  asr_d <- rbind(asr_d, di)
}

#Change 0 values in WER & CER:
asr_d<-asr_d %>%
  mutate(WER=ifelse(WER==0, 0.00001, WER))

asr_d<-asr_d %>%
  mutate(CER=ifelse(CER==0, 0.00001, CER))

asr_d$model <- factor(asr_d$model, levels = c("20", "50", "80", "100"))
contrasts(asr_d$model) <- contr.sum(4)
asr_d$Dialect <- factor(asr_d$Dialect, levels = c("balear", "nord", "nord-occidental", "valencià", "central"))
contrasts(asr_d$Dialect) <- contr.sum(5)

contrasts(asr_d$Dialect)
fit2<-glm(WER ~ model*Dialect, asr_d, family = "Gamma")
fit3<-glm(CER ~ model*Dialect, asr_d, family = "Gamma")
#Treatment contrasts for ASR: Compare everything to the Central asr in the 100 condition, because we would hypothesize that that's best. 
#asr_d$model <- factor(asr_d$model, levels = c("100", "20", "50", "80"))
contrasts(asr_d$model) <- contr.treatment(4, base=4)
contrasts(asr_d$model)

#d$dialect <- factor(asr_d$Dialect, levels = c("central", "balear", "nord", "nord-occidental", "valencià"))
contrasts(asr_d$Dialect) <- contr.treatment(5, base = 5)
contrasts(asr_d$Dialect)

fit4 <- glm(WER ~ model*Dialect, asr_d, family = "Gamma")


