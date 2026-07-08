library(tidyverse)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

patients <- read_csv("data/synthetic-patient-demographics.csv")

# ---------------------------------------------------------------------------
# First look
# ---------------------------------------------------------------------------

glimpse(patients)

# Missing values per column
patients |> summarise(across(everything(), ~sum(is.na(.))))

# Distinct value counts (useful for spotting categoricals)
patients |> summarise(across(everything(), n_distinct))

# ---------------------------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------------------------

patients |> count(gender)
patients |> count(ethnicity, sort = TRUE)
patients |> count(trial_id, sort = TRUE)

# ---------------------------------------------------------------------------
# Numeric distributions: age and BMI
# ---------------------------------------------------------------------------

patients |>
  select(age, bmi) |>
  summary()

ggplot(patients, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(title = "Age Distribution", x = "Age", y = "Count")

ggplot(patients, aes(x = bmi)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "white") +
  labs(title = "BMI Distribution", x = "BMI", y = "Count")

# ---------------------------------------------------------------------------
# Age and BMI by gender and ethnicity
# ---------------------------------------------------------------------------

ggplot(patients, aes(x = gender, y = age)) +
  geom_boxplot(fill = "steelblue") +
  labs(title = "Age by Gender", x = "Gender", y = "Age")

ggplot(patients, aes(x = ethnicity, y = bmi)) +
  geom_boxplot(fill = "steelblue") +
  coord_flip() +
  labs(title = "BMI by Ethnicity", x = "Ethnicity", y = "BMI")

# ---------------------------------------------------------------------------
# Enrollment over time
# ---------------------------------------------------------------------------

patients |>
  mutate(enrollment_month = floor_date(enrollment_date, "month")) |>
  count(enrollment_month) |>
  ggplot(aes(x = enrollment_month, y = n)) +
  geom_line(color = "steelblue") +
  labs(title = "Monthly Enrollment", x = "Month", y = "Patients Enrolled")

# ---------------------------------------------------------------------------
# Age and BMI distributions by trial (faceted)
# ---------------------------------------------------------------------------

ggplot(patients, aes(x = age, color = trial_id, fill = trial_id)) +
  geom_density(alpha = 0.15, linewidth = 0.8) +
  labs(title = "Age Distribution by Trial", x = "Age", y = "Density",
       color = "Trial", fill = "Trial")

ggplot(patients, aes(x = bmi, color = trial_id, fill = trial_id)) +
  geom_density(alpha = 0.15, linewidth = 0.8) +
  labs(title = "BMI Distribution by Trial", x = "BMI", y = "Density",
       color = "Trial", fill = "Trial")

# ---------------------------------------------------------------------------
# Patient counts by trial and site
# ---------------------------------------------------------------------------

patients |>
  count(trial_id, site_id, sort = TRUE) |>
  ggplot(aes(x = reorder(site_id, n), y = n, fill = trial_id)) +
  geom_col() +
  coord_flip() +
  labs(title = "Patients per Site (by Trial)", x = "Site", y = "Count", fill = "Trial")
