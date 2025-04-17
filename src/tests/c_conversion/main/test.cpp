#include <iostream>
#include <memory>
#include <geos/geom/GeometryFactory.h>
#include <geos/io/WKTReader.h>

int main() {
    using namespace geos::geom;
    using namespace geos::io;

    try {
        auto factory = GeometryFactory::create();
        WKTReader reader(factory.get());

        const char* raw = "POINT(1 2)";
        std::string input(raw);

        std::cout << "[WKT Input] -> " << input << "\n";

        auto geom = std::unique_ptr<Geometry>(reader.read(input));
        std::cout << "Parsed geometry type: " << geom->getGeometryType() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "GEOS failed: " << e.what() << std::endl;
    }

    return 0;
}
