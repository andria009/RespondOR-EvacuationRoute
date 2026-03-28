package id.ac.ui.respondor;

import java.io.*;
import java.net.*;
import java.util.*;

import msi.gama.precompiler.GamlAnnotations.*;
import msi.gama.precompiler.ISymbolKind;
import msi.gaml.types.IType;
import msi.gama.runtime.IScope;
import msi.gama.util.GamaListFactory;
import msi.gama.util.IList;

/**
 * GAMA Plugin: OSMDataProvider
 *
 * Provides GAML operators for querying OpenStreetMap data directly within
 * GAMA simulation models.
 *
 * Build with Maven:
 *   mvn clean package -DskipTests
 *
 * Install:
 *   Copy target/gama-respondor-plugin-*.jar to GAMA/plugins/
 *
 * Usage in GAML:
 *   list<map> villages <- osm_get_villages(bbox);
 *   list<map> shelters <- osm_get_shelters(bbox);
 *   list<map> roads    <- osm_get_roads(bbox);
 */
@SuppressWarnings("unchecked")
public class OSMDataProvider {

    private static final String OVERPASS_API =
        "https://overpass-api.de/api/interpreter";

    /**
     * GAML operator: osm_get_villages(bbox)
     * Returns list of village records: [name, lat, lon, admin_level]
     */
    @operator(
        value = "osm_get_villages",
        category = {"RespondOR", "OSM"},
        concept = {"OSM", "Villages", "Evacuation"},
        doc = @doc("Query villages from OpenStreetMap for a bounding box [south,west,north,east]")
    )
    public static IList<Map<String, Object>> osmGetVillages(
        IScope scope,
        IList<Double> bbox   // [south, west, north, east]
    ) {
        String query = buildOverpassQuery(
            "admin_level", "9",
            bbox.get(0), bbox.get(1), bbox.get(2), bbox.get(3)
        );
        return parseOverpassJSON(executeQuery(query));
    }

    /**
     * GAML operator: osm_get_shelters(bbox)
     * Returns list of shelter candidates.
     */
    @operator(
        value = "osm_get_shelters",
        category = {"RespondOR", "OSM"},
        doc = @doc("Query shelter candidates from OpenStreetMap")
    )
    public static IList<Map<String, Object>> osmGetShelters(
        IScope scope,
        IList<Double> bbox
    ) {
        String query = buildShelterQuery(
            bbox.get(0), bbox.get(1), bbox.get(2), bbox.get(3)
        );
        return parseOverpassJSON(executeQuery(query));
    }

    /**
     * GAML operator: inarisk_get_risk(lat, lon, hazard_type)
     * Returns normalized risk score [0.0 - 1.0].
     */
    @operator(
        value = "inarisk_get_risk",
        category = {"RespondOR", "InaRISK"},
        doc = @doc("Query InaRISK hazard risk score for a point. " +
                   "hazard_type: earthquake|flood|volcano|landslide")
    )
    public static Double inariskGetRisk(
        IScope scope,
        Double lat,
        Double lon,
        String hazardType
    ) {
        return InaRISKClient.queryRiskScore(lat, lon, hazardType);
    }

    // ------------------------------------------------------------------ //
    // Private helpers
    // ------------------------------------------------------------------ //

    private static String buildOverpassQuery(
        String tagKey, String tagValue,
        double south, double west, double north, double east
    ) {
        String bbox = south + "," + west + "," + north + "," + east;
        return "[out:json][timeout:60];" +
               "(" +
               "  relation[\"" + tagKey + "\"=\"" + tagValue + "\"]" +
               "    (\"boundary\"=\"administrative\")" +
               "    (" + bbox + ");" +
               "  node[\"place\"~\"village|hamlet|suburb\"]" +
               "    (" + bbox + ");" +
               ");" +
               "out center;";
    }

    private static String buildShelterQuery(
        double south, double west, double north, double east
    ) {
        String bbox = south + "," + west + "," + north + "," + east;
        return "[out:json][timeout:60];" +
               "(" +
               "  node[\"emergency\"=\"assembly_point\"](" + bbox + ");" +
               "  node[\"emergency\"=\"shelter\"](" + bbox + ");" +
               "  node[\"amenity\"=\"community_centre\"](" + bbox + ");" +
               "  node[\"amenity\"=\"school\"](" + bbox + ");" +
               "  node[\"building\"=\"public\"](" + bbox + ");" +
               ");" +
               "out body;";
    }

    private static String executeQuery(String query) {
        try {
            URL url = new URL(OVERPASS_API);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(60000);
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

            String body = "data=" + URLEncoder.encode(query, "UTF-8");
            conn.getOutputStream().write(body.getBytes("UTF-8"));

            BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream())
            );
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            return sb.toString();
        } catch (Exception e) {
            System.err.println("[OSMDataProvider] Query failed: " + e.getMessage());
            return "{}";
        }
    }

    private static IList<Map<String, Object>> parseOverpassJSON(String json) {
        IList<Map<String, Object>> result = GamaListFactory.create();

        // Simple JSON parsing (avoid external dependency)
        // In production, use org.json or Jackson
        // This is a placeholder stub
        if (json == null || json.isEmpty() || json.equals("{}")) {
            return result;
        }

        // TODO: Parse Overpass JSON elements
        // Each element: {"type":"node","id":123,"lat":-7.5,"lon":110.4,"tags":{"name":"..."}}
        // For full implementation, add org.json to pom.xml and parse properly

        return result;
    }
}
