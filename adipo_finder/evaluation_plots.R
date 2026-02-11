library(ggplot2)
library(tidyr)
library(dplyr)


# Read the data
eval_df = read.table("../../evaluation_old_vs_new.txt", sep="\t", header=TRUE)

# Reshape to long format for ggplot
eval_long = eval_df %>%
  pivot_longer(
    cols = c(P_old, R_old, F1_old, P_new, R_new, F1_new),
    names_to = c("metric", "version"),
    names_sep = "_",
    values_to = "value"
  )

# Optional: check the reshaped data
head(eval_long)
# Columns: gt_id, split, metric (P/R/F1), version (old/new), value

eval_long$version[eval_long$version == "old"] = "Previous"
eval_long$version[eval_long$version == "new"] = "Current"

# Rename metrics for readability
eval_long = eval_long %>%
  mutate(metric = recode(metric,
                         "P"  = "Precision",
                         "R"  = "Recall",
                         "F1" = "F1"))

# Set factor levels to control order in plot
eval_long$metric = factor(eval_long$metric, levels = c("Precision", "Recall", "F1"))
eval_long$version = factor(eval_long$version, levels = c("Previous", "Current"))


# Plot boxplots
p = ggplot(eval_long, aes(x = metric, y = value, fill = version)) +
  geom_boxplot(outliers = FALSE, position = position_dodge(width = 0.8)) +
  geom_point(
    color = "black",  # all points black
    position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2),
    alpha = 0.6, size = 1.5,
    show.legend = FALSE  # don't include in legend
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank()
  ) +
  scale_fill_manual(values = c("Previous" = "#FF8888", "Current" = "#3399FF")) +
  labs(x = NULL, y = "Value", fill = "Predictor", title = "All images with gt") +
  scale_y_continuous(limits = c(0, 1))

p

ggsave(filename = "../../boxplots_old_vs_new_eval_all_samp.png", 
       plot = p,          # your ggplot object
       width = 4.3,         # width in inches
       height = 3,        # height in inches
       bg = "white",
       dpi = 300)         # resolution

#now subset to test
df_test = eval_long[eval_long$split == "test",]
# Plot boxplots
p = ggplot(df_test, aes(x = metric, y = value, fill = version)) +
  geom_boxplot(outliers = FALSE, position = position_dodge(width = 0.8)) +
  geom_point(
    color = "black",  # all points black
    position = position_jitterdodge(dodge.width = 0.8, jitter.width = 0.2),
    alpha = 0.6, size = 1.5,
    show.legend = FALSE  # don't include in legend
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank()
  ) +
  scale_fill_manual(values = c("Previous" = "#FF8888", "Current" = "#3399FF")) +
  labs(x = NULL, y = "Value", fill = "Predictor") +
  scale_y_continuous(limits = c(0, 1))
p

ggsave(filename = "../../boxplots_old_vs_new_eval_test_only.png", 
       plot = p,          # your ggplot object
       width = 4.1,         # width in inches
       height = 3,        # height in inches
       bg = "white",
       dpi = 300)         # resolution

old_prec = df_test$value[df_test$metric == "Precision" & df_test$version == "Previous"]
old_prec
new_prec = df_test$value[df_test$metric == "Precision" & df_test$version == "Current"]
new_prec
t.test(new_prec, old_prec, var.equal = TRUE) #p-value = 0.8557 #ns

old_recall = df_test$value[df_test$metric == "Recall" & df_test$version == "Previous"]
old_recall
new_recall = df_test$value[df_test$metric == "Recall" & df_test$version == "Current"]
new_recall
t.test(new_recall, old_recall, var.equal = TRUE) #p-value = 0.005689

old_f1 = df_test$value[df_test$metric == "F1" & df_test$version == "Previous"]
old_f1
new_f1 = df_test$value[df_test$metric == "F1" & df_test$version == "Current"]
new_f1
t.test(new_f1, old_f1, var.equal = TRUE) #p-value = 0.01541

#now also a plot with just the new values

#now subset to test
df_test_curr = df_test[df_test$version == "Current" ,]
write_tsv(df_test_curr, "data/test_set_results.txt")
# Plot boxplots
p = ggplot(df_test_curr, aes(x = metric, y = value)) +
  geom_boxplot(outliers = FALSE, fill = "#aaaaaa") +
  geom_point(
    color = "black",  # all points black
    position = position_jitter(width = 0.2, height=0),
    alpha = 0.6, size = 1.5,
    show.legend = FALSE  # don't include in legend
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank()
  ) +
  labs(x = NULL, y = "Value") +
  scale_y_continuous(limits = c(0, 1))
p

ggsave(filename = "../../boxplot_eval_test_only.png", 
       plot = p,          # your ggplot object
       width = 2.6,         # width in inches
       height = 3,        # height in inches
       bg = "white",
       dpi = 300)         # resolution


########################
# Investigate the scaling
########################

library(tidyverse)

d = read_tsv("data/exported_feature_data.txt")
d

library(ggplot2)

plot_feature_boxplot = function(df, feature) {
  # df: your tibble
  # feature: string with the column name to plot
  
  ggplot(df, aes(x = image_id, y = .data[[feature]])) +
    geom_boxplot(outlier.size = 0.5) +
    labs(
      title = paste("Boxplot of", feature, "per image_id"),
      x = "Image ID",
      y = feature
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
    )
}

plot_feature_boxplot(d, "area") #some variation
plot_feature_boxplot(d, "eccentricity") #some variation, not dramatic
plot_feature_boxplot(d, "compactness") #ok
plot_feature_boxplot(d, "max_dist") #varies
plot_feature_boxplot(d, "distance_from_shore") #varies a lot, but that is expected
plot_feature_boxplot(d, "frac_ring_other_objects") #varies a lot
plot_feature_boxplot(d, "frac_ring_tissue") #varies a lot

#ok, now filter to only have gt = 1
unique(d$ground_truth)
dx = d[!is.na(d$ground_truth),]
d2 = dx[dx$ground_truth > 0,]
plot_feature_boxplot(d2, "area") #some variation
plot_feature_boxplot(d2, "eccentricity") #some variation, not dramatic
plot_feature_boxplot(d2, "compactness") #ok
plot_feature_boxplot(d2, "max_dist") #varies
plot_feature_boxplot(d2, "distance_from_shore") #varies a lot, but that is expected
plot_feature_boxplot(df, "frac_ring_other_objects") #varies a lot
plot_feature_boxplot(df, "frac_ring_tissue") #varies a lot

sum(is.na(d2$ground_truth))

d3 = d2[is.na(d2$area),]
d3
d4 = d2[!is.na(d2$area),]
d4
unique(d4$image_id)
unique(d2$image_id)
unique(d$image_id)

sum(is.na(d2$area))
sum(is.na(d2$compactness ))

eval_long[eval_long$metric == "F1" & eval_long$version == "Current",]
