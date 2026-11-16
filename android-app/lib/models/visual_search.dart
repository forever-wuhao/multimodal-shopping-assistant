class VisualCandidate {
  final String id, title, imageUrl, category, brand, source;
  final double? price;
  final double similarity, preferenceBonus;
  final Map<String, String> attributes;

  VisualCandidate.fromJson(Map<String, dynamic> json)
    : id = json['product_id'] as String,
      title = json['title'] as String,
      imageUrl = json['image_url'] as String,
      category = json['category'] as String,
      brand = json['brand'] as String,
      source = json['source'] as String,
      price = (json['price_cny'] as num?)?.toDouble(),
      similarity = (json['similarity'] as num).toDouble(),
      preferenceBonus = (json['preference_bonus'] as num).toDouble(),
      attributes = Map<String, String>.from(json['attributes'] as Map);
}

class VisualSearchResult {
  final List<VisualCandidate> candidates;
  final String notice;
  final String? textNote;
  final double elapsedMs;

  VisualSearchResult.fromJson(Map<String, dynamic> json)
    : candidates = (json['results'] as List)
          .map((row) => VisualCandidate.fromJson(row as Map<String, dynamic>))
          .toList(),
      notice = json['notice'] as String,
      textNote = json['text_note'] as String?,
      elapsedMs = (json['elapsed_ms'] as num).toDouble();
}
