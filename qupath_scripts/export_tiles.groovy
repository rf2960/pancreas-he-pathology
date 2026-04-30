// =====================================================
// Export CNN tiles (512x512) per class for CURRENT image
// Filename does NOT include .svs
// =====================================================

import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RectangleROI

def imageData = getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()

// === PARAMETERS ===
def tileSize = 256

// Original slide name (e.g. R4-22.svs)
def rawSlideName = server.getMetadata().getName()

// ✅ Remove .svs (or any extension)
def slideName = rawSlideName.replaceFirst(/\.[^.]+$/, "")

println "Processing slide: " + slideName

// Base output folder
def baseOutputDir = buildFilePath(PROJECT_BASE_DIR, "tiles_256", slideName)
mkdirs(baseOutputDir)

// === FIND TILE ANNOTATIONS ===
def tiles = hierarchy.getAnnotationObjects().findAll { obj ->
    def roi = obj.getROI()
    def cls = obj.getPathClass()

    roi instanceof RectangleROI &&
    Math.round(roi.getBoundsWidth())  == tileSize &&
    Math.round(roi.getBoundsHeight()) == tileSize &&
    cls != null &&
    cls.getName() != "Total area"
}

println "Found ${tiles.size()} tiles to export."

// === EXPORT ===
int i = 0

for (tile in tiles) {

    def clsName = tile.getPathClass().getName()
    def safeClassName = clsName.replaceAll("\\s+", "_")

    def classDir = buildFilePath(baseOutputDir, safeClassName)
    mkdirs(classDir)

    def roi = tile.getROI()
    int x = (int) roi.getBoundsX()
    int y = (int) roi.getBoundsY()
    int w = (int) roi.getBoundsWidth()
    int h = (int) roi.getBoundsHeight()

    def request = RegionRequest.createInstance(
            server.getPath(),
            1.0,
            x, y, w, h
    )

    // Clean filename (NO .svs)
    def filename = String.format(
            "%s_%s_tile_%05d.png",
            slideName,
            safeClassName,
            i
    )

    def outPath = buildFilePath(classDir, filename)
    writeImageRegion(server, request, outPath)

    i++
}

println "Exported ${i} tiles for slide: ${slideName}"
