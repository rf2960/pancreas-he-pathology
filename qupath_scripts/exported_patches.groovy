import qupath.lib.regions.RegionRequest
import javax.imageio.ImageIO

def imageData = getCurrentImageData()
def server = imageData.getServer()
def outputDir = buildFilePath(PROJECT_BASE_DIR, "exported_patches", getProjectEntry().getImageName())
mkdirs(outputDir)

for (annotation in getAnnotationObjects()) {
    def roi = annotation.getROI()
    if (roi == null) continue
    def pathClass = annotation.getPathClass() == null ? "Unknown" : annotation.getPathClass().getName()
    def region = RegionRequest.createInstance(server.getPath(), 1.0, roi)
    def img = server.readBufferedImage(region)

    def classDir = buildFilePath(outputDir, pathClass)
    mkdirs(classDir)
    def fileName = annotation.getName() ?: UUID.randomUUID().toString()
    def outFile = buildFilePath(classDir, fileName + ".png")

    ImageIO.write(img, "PNG", new File(outFile))
}
print "Exported all annotations for " + getProjectEntry().getImageName()
