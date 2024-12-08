import tkinter as tk
import imageio.v3 as iio
import base64
from utils import imagetobase64
class HTMLLogger:
    def __init__(self, path):
        self.path = path
        self.log = open(path, 'w', encoding="utf-8")
        self.log.write('<html><head><meta charset="utf-8"></head><body>')

    def write(self, text):
        self.log.write(text)

    def logSection(self, text):
        self.log.write('<h2 style="margin-top: 50px;">' + text + '</h2>\n')
        #self.log.flush()

    def logEntryMatch(self, name1, image1, name2, image2, alliance1=None, alliance2=None):
        #try:
        self.log.write('<p>' + name1 + (" (" + alliance1 + ")" if alliance1 is not None else "") + '<img src="data:image/png;base64,' + imagetobase64(image1) + '"> ---> '
                           + name2 + (" (" + alliance2 + ")" if alliance2 is not None else "") + '<img src="data:image/png;base64,' + imagetobase64(image2) + '"></p>\n')
        #except:
        #    print("Failed to log entry match")
        #self.log.flush()

    def logEntryStatsMatch(self, name1, stats1, name2, stats2, alliance1=None, alliance2=None):
        msg = '<p>' + name1 + (" (" + alliance1 + ")" if alliance1 is not None else "") + ' ---> ' + name2 + (" (" + alliance2 + ")" if alliance2 is not None else "")
        for stat in stats1:
            if stat in stats2:
                msg += '<br>' + stat + ': ' + str(stats1[stat].score) + ' ---> ' + str(stats2[stat].score)
            else:
                msg += '<br>' + stat + ': ' + str(stats1[stat].score) + ' ---> ' + 'None'
        for stat in stats2:
            if stat not in stats1:
                msg += '<br>' + stat + ': ' + 'None' + ' ---> ' + str(stats2[stat].score)
        msg += '</p>\n'
        self.log.write(msg)
        #self.log.flush()

    def logEntry(self, name, image, stats=None, alliance=None):
        self.log.write('<p>' + name  + (" (" + alliance + ")" if alliance is not None else "") + '<img src="data:image/png;base64,' + imagetobase64(image) + '">')
        if stats is not None:
            for stat in stats:
                self.log.write('<br>' + stat + ': ' + str(stats[stat].score))
        self.log.write('</p>\n')
        #self.log.flush()


    def close(self):
        self.log.write('</body></html>')
        self.log.close()

    def __del__(self):
        self.close()


logger = HTMLLogger('test.html')