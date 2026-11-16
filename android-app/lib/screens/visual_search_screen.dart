import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/visual_search.dart';
import '../services/api_service.dart';
import '../utils/constants.dart';

class VisualSearchScreen extends StatefulWidget {
  const VisualSearchScreen({super.key});

  @override
  State<VisualSearchScreen> createState() => _VisualSearchScreenState();
}

class _VisualSearchScreenState extends State<VisualSearchScreen> {
  final _text = TextEditingController();
  final _budget = TextEditingController();
  final _category = TextEditingController();
  final _brand = TextEditingController();
  final _color = TextEditingController();
  final _size = TextEditingController();
  final Set<String> _selected = {};
  File? _image;
  VisualSearchResult? _result;
  String? _error;
  bool _busy = false, _preferLowPrice = false;
  bool? _ready;

  @override
  void initState() {
    super.initState();
    _checkStatus();
  }

  Future<void> _checkStatus() async {
    try {
      final ready = await ApiService().visualReady();
      if (mounted)
        setState(() {
          _ready = ready;
          _error = null;
        });
    } catch (_) {
      if (mounted) setState(() => _error = '无法连接检索服务，请检查后端地址后重试。');
    }
  }

  @override
  void dispose() {
    for (final controller in [
      _text,
      _budget,
      _category,
      _brand,
      _color,
      _size,
    ]) {
      controller.dispose();
    }
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    try {
      final picked = await ImagePicker().pickImage(
        source: source,
        maxWidth: 1600,
        imageQuality: 90,
      );
      if (picked != null && mounted) {
        setState(() {
          _image = File(picked.path);
          _result = null;
          _selected.clear();
        });
      }
    } catch (_) {
      if (mounted) setState(() => _error = '无法获取照片，请检查相机或相册权限。');
    }
  }

  Future<void> _search() async {
    final budgetText = _budget.text.trim();
    final budget = double.tryParse(budgetText);
    if (budgetText.isNotEmpty &&
        (budget == null || !budget.isFinite || budget < 0)) {
      setState(() => _error = '请输入有效的预算金额');
      return;
    }
    if (_image == null && _text.text.trim().isEmpty) {
      setState(() => _error = '先拍照、选择图片或输入商品描述');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _selected.clear();
    });
    try {
      final result = await ApiService().visualSearch(
        image: _image,
        text: _text.text.trim(),
        category: _category.text.trim(),
        brand: _brand.text.trim(),
        maxPrice: budget,
        attributes: {
          if (_color.text.trim().isNotEmpty) 'color': _color.text.trim(),
          if (_size.text.trim().isNotEmpty) 'size': _size.text.trim(),
        },
        preferLowPrice: _preferLowPrice,
      );
      if (mounted)
        setState(() {
          _result = result;
          _ready = true;
        });
    } catch (error) {
      if (mounted)
        setState(
          () => _error = error is ApiException ? error.message : '检索失败，请稍后重试',
        );
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Widget _field(
    TextEditingController controller,
    String label, {
    bool number = false,
  }) => Padding(
    padding: const EdgeInsets.only(bottom: 12),
    child: TextField(
      controller: controller,
      enabled: !_busy,
      onChanged: (_) => setState(() {
        _result = null;
        _selected.clear();
      }),
      keyboardType: number
          ? const TextInputType.numberWithOptions(decimal: true)
          : TextInputType.text,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
    ),
  );

  void _compare() {
    final items = _result!.candidates
        .where((item) => _selected.contains(item.id))
        .toList();
    final keys = items.expand((item) => item.attributes.keys).toSet().toList()
      ..sort();
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('候选商品对比'),
        content: SingleChildScrollView(
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: DataTable(
              columns: [
                const DataColumn(label: Text('项目')),
                ...items.map(
                  (item) => DataColumn(
                    label: SizedBox(width: 150, child: Text(item.title)),
                  ),
                ),
              ],
              rows: [
                DataRow(
                  cells: [
                    const DataCell(Text('资料价格')),
                    ...items.map(
                      (item) => DataCell(
                        Text(item.price == null ? '未提供' : '¥${item.price}'),
                      ),
                    ),
                  ],
                ),
                for (final key in keys)
                  DataRow(
                    cells: [
                      DataCell(Text(key)),
                      ...items.map(
                        (item) => DataCell(Text(item.attributes[key] ?? '未提供')),
                      ),
                    ],
                  ),
                DataRow(
                  cells: [
                    const DataCell(Text('资料来源')),
                    ...items.map(
                      (item) => DataCell(
                        SizedBox(
                          width: 150,
                          child: SelectableText(item.source),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('关闭'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('拍照找同款')),
    body: ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Text(
          '找到相似商品，再按预算和参数筛选。',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        if (_ready == false)
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  const Text('商品图库尚未准备好。导入商品照片后即可检索。'),
                  TextButton(
                    onPressed: _checkStatus,
                    child: const Text('重新检查'),
                  ),
                ],
              ),
            ),
          ),
        if (_image != null)
          Stack(
            alignment: Alignment.topRight,
            children: [
              Center(
                child: Image.file(_image!, height: 180, fit: BoxFit.contain),
              ),
              IconButton(
                onPressed: _busy
                    ? null
                    : () => setState(() {
                        _image = null;
                        _result = null;
                      }),
                icon: const Icon(Icons.close),
                tooltip: '移除照片',
              ),
            ],
          ),
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _busy ? null : () => _pick(ImageSource.camera),
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('拍照'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: OutlinedButton.icon(
                onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                icon: const Icon(Icons.photo_outlined),
                label: const Text('相册'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        _field(_text, '商品描述（可选，当前模型建议英文）'),
        ExpansionTile(
          title: const Text('预算与商品条件'),
          initiallyExpanded: true,
          children: [
            _field(_budget, '最高预算（元）', number: true),
            _field(_category, '品类（与商品资料一致）'),
            _field(_brand, '品牌'),
            _field(_color, '颜色'),
            _field(_size, '尺寸'),
            SwitchListTile(
              title: const Text('相似候选中偏好低价'),
              value: _preferLowPrice,
              onChanged: _busy
                  ? null
                  : (value) => setState(() {
                      _preferLowPrice = value;
                      _result = null;
                      _selected.clear();
                    }),
            ),
          ],
        ),
        FilledButton(
          onPressed: _busy || _ready == false ? null : _search,
          child: Text(_busy ? '正在检索…' : '查找商品'),
        ),
        if (_busy) const LinearProgressIndicator(),
        if (_error != null)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 12),
            child: Text(
              _error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ),
        if (_result != null) ...[
          const SizedBox(height: 12),
          Text(_result!.notice),
          if (_result!.textNote != null) Text(_result!.textNote!),
          Text(
            '${_result!.candidates.length} 个候选 · ${_result!.elapsedMs.toStringAsFixed(0)} ms',
          ),
          if (_result!.candidates.isEmpty)
            const Padding(
              padding: EdgeInsets.all(20),
              child: Text('没有满足全部条件的商品。可修改条件后再次查询，系统不会自动放宽预算。'),
            ),
          ..._result!.candidates.map(
            (item) => Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Image.network(
                          '${Constants.apiBaseUrl}${item.imageUrl}',
                          width: 72,
                          height: 72,
                          fit: BoxFit.contain,
                          errorBuilder: (_, __, ___) => const SizedBox(
                            width: 72,
                            child: Icon(Icons.image_not_supported),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            item.title,
                            style: const TextStyle(fontWeight: FontWeight.bold),
                          ),
                        ),
                        Checkbox(
                          value: _selected.contains(item.id),
                          onChanged:
                              !_selected.contains(item.id) &&
                                  _selected.length >= 3
                              ? null
                              : (selected) => setState(() {
                                  if (selected == true &&
                                      _selected.length < 3) {
                                    _selected.add(item.id);
                                  } else {
                                    _selected.remove(item.id);
                                  }
                                }),
                        ),
                      ],
                    ),
                    Text(item.price == null ? '价格未提供' : '资料价格 ¥${item.price}'),
                    Text(
                      '相似度 ${item.similarity.toStringAsFixed(3)}${item.preferenceBonus > 0 ? ' · 已应用低价偏好' : ''}',
                    ),
                    if (item.attributes.isNotEmpty)
                      Text(
                        item.attributes.entries
                            .map((e) => '${e.key}: ${e.value}')
                            .join(' · '),
                      ),
                    SelectableText('来源：${item.source}'),
                  ],
                ),
              ),
            ),
          ),
          OutlinedButton(
            onPressed: _selected.length >= 2 ? _compare : null,
            child: Text('对比已选商品（${_selected.length}/3）'),
          ),
        ],
      ],
    ),
  );
}
