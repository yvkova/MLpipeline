# Plotting results

# import libraries
import matplotlib.pyplot as plt
import re


# plot all scores to a barchart diagram
def plot_bar_chart(tts_results, cv_scores, gcv_scores):
    fig5, axs5 = plt.subplots(ncols=len(tts_results))

    for key, value in tts_results.items():
        axs5[0].bar(re.sub(r'(?=[A-Z])', '\n', key), value)
        axs5[0].set_title("TTS")

    for key, value in cv_scores.items():
        axs5[1].bar(re.sub(r'(?=[A-Z])', '\n', key), value.mean())
        axs5[1].set_title("CV")

    for key, value in gcv_scores.items():
        axs5[2].bar(re.sub(r'(?=[A-Z])', '\n', key), value)
        axs5[2].set_title("GCV")
