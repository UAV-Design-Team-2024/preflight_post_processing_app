#include <geos_c.h>
#include <iostream>

int main() {
    GEOSContextHandle_t ctx = GEOS_init_r();

    const char* wkt = "LINESTRING (0 0, 1 1)";
    GEOSWKTReader* reader = GEOSWKTReader_create_r(ctx);

    GEOSGeometry* geom = GEOSWKTReader_read_r(ctx, reader, wkt);
    if (!geom) {
        std::cerr << "GEOS C API failed to parse WKT!\n";
    } else {
        std::cout << "WKT parsed successfully via GEOS C API\n";
        GEOSGeom_destroy_r(ctx, geom);
    }

    GEOSWKTReader_destroy_r(ctx, reader);
    GEOS_finish_r(ctx);
    return 0;
}
