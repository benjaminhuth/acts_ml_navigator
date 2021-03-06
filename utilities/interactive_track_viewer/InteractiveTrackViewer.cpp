#include "InteractiveTrackViewer.hpp"

#include <QAction>
#include <QChart>
#include <QChartView>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLegendMarker>
#include <QListWidget>
#include <QMenuBar>
#include <QMessageBox>
#include <QScatterSeries>
#include <QToolBar>
#include <QVBoxLayout>
#include <QStatusBar>
#include <QDesktopWidget>

#include <fstream>

#include <spdlog/spdlog.h>

#include "TrackListWidgetItem.hpp"
#include "SeriesSelectDialog.hpp"
#include "csv.hpp"

const std::vector<std::pair<const char *, QColor>> series_properties = {
    {"in1", QColor("#006600")},   // Dark green
    {"in2", QColor("#00cc00")},   // Light green
    {"in3", QColor("#ffff00")},   // Yellow
    {"in5", QColor("#ff3300")},   // Bright red
    {"in10", QColor("#800000")},  // Dark red
    {"other", QColor("#737373")}, // Gray
};

void append_to_series_vector(auto row_tuple_container,
                             std::vector<QtCharts::QScatterSeries *> series_vec)
{
    for( const auto &[pos, r, z, score] : row_tuple_container )
    {
        const QPointF p(static_cast<double>(z), static_cast<double>(r));

        if( score == 0 )
            series_vec[0]->append(p);
        else if( score == 1 )
            series_vec[1]->append(p);
        else if( score == 3 )
            series_vec[2]->append(p);
        else if( score < 5 )
            series_vec[3]->append(p);
        else if( score < 10 )
            series_vec[4]->append(p);
        else
            series_vec[5]->append(p);
    }
}

InteractiveTrackViewer::InteractiveTrackViewer(QWidget *parent) :
    QMainWindow(parent),
    m_rnd_gen(std::random_device{}()),
    m_chart_view(new QtCharts::QChartView(this)),
    m_chart(m_chart_view->chart())
{
    
    resize(QDesktopWidget().availableGeometry(this).size() * 0.7);
    // UI
    auto central_layout = new QHBoxLayout(this);

    m_chart_view->setRenderHint(QPainter::Antialiasing);
    m_chart_view->setRubberBand(QtCharts::QChartView::RectangleRubberBand);
    central_layout->addWidget(m_chart_view, 3);

    m_list_widget = new QListWidget(this);
    m_list_widget->setSortingEnabled(true);
    m_list_widget->sortItems(Qt::SortOrder::DescendingOrder);
    central_layout->addWidget(m_list_widget, 1);

    setCentralWidget(new QWidget(this));
    centralWidget()->setLayout(central_layout);

    statusBar()->showMessage("Nothing loaded yet");
    
    SPDLOG_DEBUG("Created UI");

    // Actions
    auto open_action = new QAction("Open", this);
    connect(open_action, &QAction::triggered, this,
            &InteractiveTrackViewer::open);

    auto sort_action = new QAction("Tracks Sort Order", this);
    connect(
        sort_action, &QAction::triggered, this,
        [&, order = static_cast<bool>(Qt::SortOrder::DescendingOrder)]() mutable
        {
            order = !order;
            m_list_widget->sortItems(static_cast<Qt::SortOrder>(order));
        });
    
    auto reset_zoom_action = new QAction("Reset zoom", this);
    connect(reset_zoom_action, &QAction::triggered, m_chart, &QtCharts::QChart::zoomReset);
    
    auto select_series_action = new QAction("Select visible points", this);
    connect(select_series_action, &QAction::triggered, this, [&]()
    {
        SeriesSelectDialog dialog(m_chart->series());
        dialog.exec();
    });

    QToolBar *file_tool_bar = addToolBar("Main Toolbar");
    file_tool_bar->addAction(open_action);
    file_tool_bar->addAction(sort_action);
    file_tool_bar->addAction(reset_zoom_action);
    file_tool_bar->addAction(select_series_action);

    file_tool_bar->setToolButtonStyle(Qt::ToolButtonTextOnly);
    file_tool_bar->setMovable(false);

    // Init current track series
    for( const auto &[name, color] : series_properties )
    {
        auto series = new QtCharts::QScatterSeries();
        series->setColor(color);
        series->setPen(QPen(QBrush(), 5.));
        series->setBorderColor(Qt::black);
        series->setMarkerSize(20.);

        m_current_track.push_back(series);
    }

    // Select Track
    connect(m_list_widget, &QListWidget::itemActivated, this,
            [&](auto item)
            {
                auto item_ptr = dynamic_cast<TrackListWidgetItem *>(item);

                if( !item_ptr )
                    return;
                
                display_track(item_ptr->track->second);
            });
}

InteractiveTrackViewer::~InteractiveTrackViewer() {}

void InteractiveTrackViewer::display_track(const std::span<RowTuple> &track) 
{
    // Remove old series
    for( auto series : m_current_track )
    {
        m_chart->removeSeries(series);
        series->clear();
    }

    // Fill in data
    append_to_series_vector(track, m_current_track);

    // Add new series
    for( auto series : m_current_track )
    {
        m_chart->addSeries(series);
        series->attachAxis(m_chart->axes()[0]);
        series->attachAxis(m_chart->axes()[1]);
        m_chart->legend()->markers(series).at(0)->setVisible(false);
    }
}

void InteractiveTrackViewer::open()
{    
    QString filename = QFileDialog::getOpenFileName(this, "Open the file");
//     QString filename =
//         "/home/benjamin/Dokumente/acts_project/ml_navigator/data/models/"
//         "target_pred_navigator_pre/generic/20210221-170851-emb10-acc1-nn.csv";

    std::ifstream file(filename.toStdString());

    if( !file.is_open() )
    {
        QMessageBox::warning(
            this, "Warning",
            fmt::format("Cannot open file '{}'", filename.toStdString())
                .c_str());
        return;
    }
    
    // Clear everything
    m_chart->removeAllSeries();
    
    for( auto axis : m_chart->axes() )
        m_chart->removeAxis(axis);
    
    m_data.clear();
    m_tracks.clear();

    setWindowTitle(fmt::format("Interactive Track Viewer ({})", filename.toStdString()).c_str());
    
    try
    {
        bool header = true;
        for( const auto &row : CSVRange(file) )
        {
            if( header )
            {
                header = false;
                continue;
            }

            auto track_pos = std::stoi(row[1]);
            auto r       = std::stod(row[2]);
            auto z       = std::stod(row[3]);
            auto score     = std::stoi(row[4]);

            m_data.push_back({track_pos, r, z, score});
        }
    }
    catch( std::exception &e )
    {
        QMessageBox::warning(
            this, "Warning",
            fmt::format("Error importing file '{}'", e.what()).c_str());
        return;
    }

    SPDLOG_DEBUG("Imported CSV");

    // Find tracks
    auto start = m_data.begin();
    for( auto it = m_data.begin(); it != m_data.end(); ++it )
    {
        auto next = std::next(it);

        if( next == m_data.end() || std::get<0>(*next) == 0 )
        {
            std::span<RowTuple> track(start, next);
            auto mean_score =
                static_cast<double>(std::accumulate(
                    track.begin(), track.end(), 0,
                    [](auto a, auto b) { return a + std::get<3>(b); })) /
                static_cast<double>(track.size());

            m_tracks.push_back({mean_score, track});

            start = next;
        }
    }
    
    // Display on status bar
    statusBar()->showMessage(fmt::format("Loaded {} tracks (in total {} points)", m_tracks.size(), m_data.size()).c_str());

    // Add to widget
    for( auto it = m_tracks.cbegin(); it != m_tracks.cend(); ++it )
    {
        auto item = new TrackListWidgetItem(&*it, m_list_widget);
        item->setText(fmt::format("track length: {} \t mean score: {:.2f}",
                                  it->second.size(), it->first)
                          .c_str());
    }

    // Create background data
    auto background_data = m_data;
    std::ranges::shuffle(background_data, m_rnd_gen);
    background_data.resize(std::min(m_num_background_data, m_data.size()));

    // Create background serieses
    std::vector<QtCharts::QScatterSeries *> series_vec;

    for( const auto &[name, color] : series_properties )
    {
        auto series = new QtCharts::QScatterSeries();
        series->setName(name);
        series->setColor(color);
        series->setBorderColor(Qt::transparent);
        series->setMarkerSize(10.);

        connect(series, &QtCharts::QXYSeries::clicked, this,
                [&](auto point)
                {
                    auto found = std::ranges::find_if(
                        m_data,
                        [&](auto a)
                        {
                            return std::get<2>(a) == point.x() &&
                                   std::get<1>(a) == point.y();
                        });
                    
                    if( found == m_data.end() )
                        return;

                    for(const auto &[score, track] : m_tracks)
                    {
                        auto found_ptr = &*found;
                        
                        if( found_ptr >= &*track.begin() && found_ptr < &*track.end() )
                        {
                            display_track(track);
                            return;
                        }
                    }
                });

        series_vec.push_back(series);
    }

    append_to_series_vector(background_data, series_vec);

    SPDLOG_DEBUG("Created Series");

    // Add series to chart
    for( auto series : series_vec )
        m_chart->addSeries(series);

    SPDLOG_DEBUG("Added Series");

    // Set plot plot range
    auto max_r = std::get<1>(
        *std::ranges::max_element(background_data, [](auto a, auto b)
                                  { return std::get<1>(a) < std::get<1>(b); }));
    auto max_z = std::get<2>(
        *std::ranges::max_element(background_data, [](auto a, auto b)
                                  { return std::get<2>(a) < std::get<2>(b); }));

    SPDLOG_DEBUG("max r: {}, max z: {}", max_r, max_z);

    max_r *= 1.1;
    max_z *= 1.1;

    m_chart->createDefaultAxes();
    m_chart->axes()[0]->setRange(-max_z, max_z);
    m_chart->axes()[1]->setRange(0, max_r);

    SPDLOG_DEBUG("Finished importing");
}

void InteractiveTrackViewer::exit() {}
