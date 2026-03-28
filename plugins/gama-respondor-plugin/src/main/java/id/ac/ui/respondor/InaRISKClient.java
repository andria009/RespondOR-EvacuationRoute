package id.ac.ui.respondor;

import java.io.*;
import java.net.*;

/**
 * InaRISK API client for GAMA plugin.
 * Queries BNPB InaRISK hazard risk index service.
 */
public class InaRISKClient {

    private static final String BASE_URL =
        "https://gis.bnpb.go.id/server/rest/services/inarisk";

    private static final java.util.Map<String, String> SERVICES =
        new java.util.HashMap<String, String>() {{
            put("earthquake", "INDEKS_BAHAYA_GEMPABUMI");
            put("flood",      "INDEKS_BAHAYA_BANJIR");
            put("volcano",    "INDEKS_BAHAYA_GUNUNGAPI");
            put("landslide",  "INDEKS_BAHAYA_TANAHLONGSOR");
        }};

    /**
     * Query risk score for a point.
     * @param lat latitude (WGS84)
     * @param lon longitude (WGS84)
     * @param hazardType earthquake|flood|volcano|landslide
     * @return normalized risk score [0.0 - 1.0]
     */
    public static double queryRiskScore(double lat, double lon, String hazardType) {
        String service = SERVICES.getOrDefault(hazardType.toLowerCase(), null);
        if (service == null) {
            return 0.0;
        }

        double[] mercator = toWebMercator(lat, lon);
        double x = mercator[0];
        double y = mercator[1];

        String urlStr = BASE_URL + "/" + service + "/MapServer/0/query"
            + "?geometry=" + x + "," + y
            + "&geometryType=esriGeometryPoint"
            + "&inSR=102100"
            + "&spatialRel=esriSpatialRelIntersects"
            + "&outFields=INDEKS_BAHAYA"
            + "&outSR=4326"
            + "&returnGeometry=false"
            + "&resultRecordCount=1"
            + "&where=1=1"
            + "&f=json";

        try {
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(30000);

            BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream())
            );
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }

            return parseRiskFromJSON(sb.toString());
        } catch (Exception e) {
            System.err.println("[InaRISKClient] Query failed: " + e.getMessage());
            return 0.0;
        }
    }

    private static double parseRiskFromJSON(String json) {
        // Simple extraction of INDEKS_BAHAYA value
        // Pattern: "INDEKS_BAHAYA":X.X
        try {
            int idx = json.indexOf("\"INDEKS_BAHAYA\"");
            if (idx < 0) return 0.0;
            int colon = json.indexOf(":", idx);
            int end = json.indexOf(",", colon);
            if (end < 0) end = json.indexOf("}", colon);
            String valStr = json.substring(colon + 1, end).trim();
            double rawVal = Double.parseDouble(valStr);
            // Normalize from 1-3 scale to 0-1
            if (rawVal > 1.0) {
                rawVal = (rawVal - 1.0) / 2.0;
            }
            return Math.max(0.0, Math.min(1.0, rawVal));
        } catch (Exception e) {
            return 0.0;
        }
    }

    private static double[] toWebMercator(double lat, double lon) {
        double x = lon * 20037508.34 / 180.0;
        double y = Math.log(Math.tan((90.0 + lat) * Math.PI / 360.0))
                   / (Math.PI / 180.0);
        y = y * 20037508.34 / 180.0;
        return new double[]{x, y};
    }
}
