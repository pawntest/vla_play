# 開発引き継ぎ資料(so101-tool)

このドキュメント1枚で開発を引き継げることを目的とする。ユーザー向け情報は README、
各論は docs/ の他ファイルにあり、ここでは**構造・規約・ハマりどころ**だけを扱う。

## 1. プロジェクトの現状

LeRobot SO-101 用の統合ツール(ベータ 0.2.0b1)。シム(MuJoCo)・実機・SSHリモートを
同一UIで扱い、シーン定義 → テレオペ/自動デモ収集 → 学習(lerobot-train)→ ポリシー実行
まで完結する。UI は日本語。未実装機能の一覧は本書 §7。

## 2. 環境(重要)

- **venv が2つある**: `.venv`(Python 3.11、lerobot なし、開発の主戦場)と
  `.venv312`(Python 3.12 + lerobot 0.6 + torch CPU、実機/学習系のテスト用)。
  pyproject の requires-python は >=3.12(uv 管理)だが、コア機能は 3.10+ で動く。
- テストは**両方**で回す: `.venv/bin/pytest` と `.venv312/bin/pytest`。lint は
  `.venv/bin/ruff check src tests`。
- ヘッドレス描画: `MUJOCO_GL` は **mujoco を import する前**に設定必須。
  `config.ensure_headless_gl()` が担い、`cli.main()` と `tests/conftest.py` が呼ぶ。
  この環境は OSMesa(EGL なし)。
- ブラウザ検証は Playwright + `/opt/pw-browsers/chromium`(スクリーンショットで確認する
  文化。tests だけでは UI の壊れは検出できない)。

## 3. 全体構造(データの流れ)

```
YAML (scenario.py) ──build_model──▶ MjSpec+flexcomp ──▶ mujoco.MjModel
                                                          │
CLI/GUI/NL/API ──Command──▶ ControlLoop(50Hz thread) ──▶ RobotInterface backend
                               │ LoopSnapshot(immutable)      │
viser UI (app.py render loop 30Hz) ◀──────────────────────────┘
```

- **正準単位**(全内部コードで統一): アーム `q` は `(5,)` ラジアン
  `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`、
  グリッパーは 0..1。度数(lerobot)や 0-100 への変換は**バックエンド境界のみ**
  (`config.JointMap`)。
- **ControlLoop**(`control/loop.py`)がロボットに触れる唯一のスレッド。入力は
  Command キュー、出力は immutable な LoopSnapshot。e-stop はラッチ式。
  全書き込みは `control/safety.py`(リミット・速度・床面)を通る。
- **Kinematics**(`kinematics.py`)は**スレッド非安全**。スレッドごとに1インスタンス
  (ループ用・GUI用・テスト用を分けるのが規約)。IK は mink(soft orientation 0.3、
  5自由度のため姿勢は best-effort)。ジョグは `plan_linear` で**動く前に全経路検証**、
  到達不能なら一切動かない契約。

## 4. モジュール早見表

| ファイル | 役割・要点 |
| --- | --- |
| `api.py` | 公開 Python API。`Session` がバックエンド選択(シム/物理/実機/Linked)+ループ起動。CLI/GUI の土台 |
| `config.py` | 単位・リミット・JointMap・AppConfig。`ensure_headless_gl` |
| `scenario.py` | YAML→MjSpec。布(flexcomp、**ネイティブ elasticity**、プラグイン不使用)、カメラ(固定/アーム取付)、環境。布は `edge equality` と併用不可 |
| `robot/base.py` | RobotInterface + RobotState(qpos_full はシーン系のみ) |
| `robot/sim.py` | 運動学シム(注入可能クロック。テストは FakeClock で決定論化) |
| `robot/physics_sim.py` | mj_step 物理シム + オフスクリーンカメラ描画 |
| `robot/lerobot_backend.py` | 実機(SO101Follower、度数変換はここだけ)。接続失敗時は _robot=None に戻す |
| `robot/linked.py` | **実機↔シム連動の中核**。to_sim/to_real/both、実機側は常にバックグラウンド接続・失敗時はシム縮退(`real_status` に理由)。reconnect_real() で再試行 |
| `teleop/receiver.py` | SSH テレオペ受信(127.0.0.1 固定 + トークン、全二重)。RemoteArmBackend が「トンネル越しの実機」を RobotInterface 化 |
| `teleop/client.py` | 手元側クライアント(leader/follower/sine)。follower は下り target_q を実機に適用 |
| `control/loop.py` | 50Hz ループ・モード(IDLE/MIRROR/RULE/POLICY/TELEOP)・プリミティブ実行器 |
| `viz/robot_view.py` | mjModel→viser メッシュ。カメラフラスタム描画。**mesh_face はメッシュローカル添字**(vertadr を引くと壊れる、既修正) |
| `viz/panel.py` | サイドバー全部(状態バナー・タブ・モード自動切替・カメラ表示モード・実機エラーヒント) |
| `viz/direct_drag.py` | 3Dメッシュ直接ドラッグ(物体移動・アーム誘導) |
| `viz/overlay_server.py` | メインUI: viser を iframe 埋め込み+カメラ縦列オーバーレイのラッパーページ(viser_port+1。viser は画面固定DOMを持てないための構成) |
| `nl/agent.py` | Claude tool-use → 動作プリミティブ(ANTHROPIC_API_KEY 時のみ) |
| `policy/runner.py` | lerobot ポリシー推論。**正式パイプライン必須**: build_dataset_frame → prepare_observation_for_inference → preprocess → select_action → postprocess → make_robot_action |
| `data/recorder.py`, `demo/` | LeRobotDataset 録画・スクリプトデモ生成(FakeClock で決定論、finalize() 必須) |
| `training.py` | lerobot-train ラッパー(draccus CLI、`--policy.push_to_hub=false` 必須)+ Colab notebook 出力。`_find_train_exe` は sys.executable 隣を優先 |
| `cli.py` | tyro サブコマンド: run / doctor(実機自己診断)/ teleop-client / scripted-demos / train / ik-check。ロギング設定とポート監視スレッドもここ |

## 5. ハマりどころ(実際に踏んだもの)

- viser `gui.reset()` は**タブグループがあるとクラッシュ** → `panel.cleanup_gui()` で
  タブ→グループの順に remove してから reset(app._attach 参照)
- viser `scene.reset()` はデフォルト照明も消す → RobotView はハンドル追跡して個別 remove
  (子→親の逆順)
- MoveL/ジョグの完了判定は**実測値**で行う(safety filter が指令をレート制限するため、
  指令値判定だと未達で完了する)
- tyro の list[str] はダッシュ始まり値を受けられない → TrainArgs.extra は文字列
  (shlex.split)
- プロキシ環境: pytorch.org 直リンクは 403 → PyPI から。ACT の事前学習バックボーンDLも
  遮断 → スモークは `--policy.pretrained_backbone_weights=null`
- 実機の接続失敗は**アプリを殺さない**のが不変条件(LinkedBackend が吸収)。新しい
  失敗モードを足すときは `viz/panel._real_error_hint` に日本語ヒントも足す
- テレオペの全二重は**1ソケット1リーダー**(makefile を共有。2つ作ると壊れる)
- スモークテスト(test_smoke_headless)はマシン負荷でまれにタイムアウトする
  (フリーポート使用済み。単体再実行で緑なら環境要因)

## 6. 検証の型(このリポジトリの流儀)

1. `.venv/bin/ruff check --fix src tests`
2. `.venv/bin/pytest -q` と `.venv312/bin/pytest -q`(計 ~110 テスト、全緑維持)
3. UI を触った変更は必ず実起動+Playwright スクリーンショットで目視
   (`so101-tool run --scenario examples/pick_cube.yaml --viser-port 809x` →
   `page.goto` → `inner_text`/screenshot で断言)
4. コミットは機能単位で日本語 or 英語の要約 + 本文。ユーザーも同ブランチに push する
   ため **push 前に必ず fetch+rebase**(force push 禁止)

## 7. 未実装・次の課題(優先順)

1. 実機USBカメラの GUI/CLI 対応(現状 `AppConfig.cameras` 辞書のみ — 実機録画の本線が塞がっている)
2. ポリシー自動評価(N回実行→成功率。`demo/scripted.py` の成功判定を流用可)
3. 衝突チェック(現状は床面のみ)・カスタムキープアウト
4. キャリブレーション(JointMap)の GUI 編集+設定ファイル永続化(現状コード編集)
5. データセットブラウザ / 録画自動化(成功判定・自動リセット)/ 損失曲線表示
6. 他ロボット対応(URDF/MJCF 取り込み)・台形速度プロファイル・軌道プレビュー
7. PyPI/Docker 配布、診断テレメトリ、シーンライブラリ管理

## 8. 資料マップ

README(セットアップ・起動・トラブルシュート)/ docs/architecture.md(座標系・
スレッド詳細)/ docs/scenario_reference.md(YAML 全仕様)/ docs/data_and_training.md
(収集→学習)/ docs/hardware.md(実機安全手順)/ docs/teleop_remote.md(リモート
接続のセキュリティ設計)。シーン例は examples/ の2つ。アームモデルは
mujoco_menagerie の robotstudio_so101(Apache-2.0)を assets に同梱。
