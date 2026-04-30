import qupath.lib.regions.RegionRequest
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RectangleROI

// === 1. PARAMETERS ===
def tileSize = 256
def overlapThreshold = 0.8
def totalClassName = 'Total area'
def defaultClassName = 'Other' // Everything in 'Total area' not otherwise labeled

def imageData = getCurrentImageData()
def server = imageData.getServer()
def hierarchy = imageData.getHierarchy()
def slideName = server.getMetadata().getName().replaceFirst(/\.[^.]+$/, "")

// Setup Output Directory
def baseOutputDir = buildFilePath(PROJECT_BASE_DIR, "Spatial_Tiles_256", slideName)
mkdirs(baseOutputDir)

// === 2. GATHER ANNOTATIONS ===
def allAnnotations = hierarchy.getAnnotationObjects()
def totals = allAnnotations.findAll { it.getPathClass()?.getName() == totalClassName }
def labeledRegions = allAnnotations.findAll { it.getPathClass() && it.getPathClass().getName() != totalClassName }

if (totals.isEmpty()) {
    print "ERROR: No 'Total area' found. Please draw a boundary around your tissue."
    return
}

// === SPATIAL TILE GENERATION & EXPORT ===
int count = 0
totals.each { total ->
    def roi = total.getROI()
    double startX = roi.getBoundsX()
    double startY = roi.getBoundsY()
    double width = roi.getBoundsWidth()
    double height = roi.getBoundsHeight()

    for (double y = startY; y < startY + height; y += tileSize) {
        for (double x = startX; x < startX + width; x += tileSize) {
            
            // Define the tile geometry
            def tileRoi = ROIs.createRectangleROI(x, y, tileSize, tileSize, roi.getImagePlane())
            
            // Optimization: Only process tiles where the center is inside the tissue boundary
            if (!roi.contains(x + tileSize/2, y + tileSize/2)) continue

            // Identify the Class by checking overlap with your manual annotations
            def clsName = defaultClassName
            def tileShape = tileRoi.getShape()
            def tileArea = tileRoi.getArea()

            for (anno in labeledRegions) {
                def aShape = anno.getROI().getShape()
                def intersection = new java.awt.geom.Area(tileShape)
                intersection.intersect(new java.awt.geom.Area(aShape))
                def intersectBounds = intersection.getBounds2D()
                def overlapArea = intersectBounds.width * intersectBounds.height
                
                if ((overlapArea / tileArea) >= overlapThreshold) {
                    clsName = anno.getPathClass().getName()
                    break
                }
            }

            // --- THE SPATIAL NAMING CONVENTION ---
            // Format: SlideID_ClassName_[x=123,y=456].png
            def safeClassName = clsName.replaceAll("\\s+", "_")
            def filename = String.format("%s_%s_[x=%d,y=%d].png", slideName, safeClassName, (int)x, (int)y)
            
            // Optional: Create an annotation in QuPath so you can see the grid
            def tileAnno = PathObjects.createAnnotationObject(tileRoi, PathClass.fromString(clsName))
            tileAnno.setName(filename)
            hierarchy.addObjects([tileAnno])

            // Export the tile image
            def classDir = buildFilePath(baseOutputDir, safeClassName)
            mkdirs(classDir)
            def request = RegionRequest.createInstance(server.getPath(), 1.0, (int)x, (int)y, tileSize, tileSize)
            writeImageRegion(server, request, buildFilePath(classDir, filename))
            
            count++
        }
    }
}

fireHierarchyUpdate()
print "SUCCESS: Exported ${count} spatial tiles to: ${baseOutputDir}"