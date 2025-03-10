/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#pragma once

#include <iostream>
#include <ostream>
#include <string>

//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#ifdef ERROR
#undef ERROR
#endif

namespace mpc
{
    /**
     * @brief Basic logger system
     */
    class Logger
    {
    public:
        /**
         * @brief Internal log message enumerator
         * 
         */
        enum log_type
        {
            DETAIL = 0,
            INFO = 1,
            ERROR = 2
        };

        /**
         * @brief External desired log level enumerator
         */
        enum log_level
        {
            DEEP = 0,
            NORMAL = 1,
            ALERT = 2,
            NONE
        };

        /**
         * @brief Return the instance of the logger
         * 
         * @return Logger& logger instance
         */
        static Logger &instance()
        {
            static Logger instance;
            return instance;
        }

        /**
         * @brief Reset the logger configuration
         * 
         * @return Logger& logger instance
         */
        Logger &reset()
        {
            resetImpl();
            return *this;
        }

        /**
         * @brief Define the log type for the following message
         * 
         * @param type level type enumerator
         * @return Logger& logger instance
         */
        Logger &log(log_type type)
        {
            Logger::instance().currentType = type;
            if ((int)Logger::instance().thresholdLevel <= (int)Logger::instance().currentType) {
                *(Logger::instance().os) << "[MPC++";
                if (!Logger::instance().prefix.empty())
                {
                    *(Logger::instance().os) << " " << Logger::instance().prefix << "] ";
                }
                else
                {
                    *(Logger::instance().os) << "] ";
                }
            }

            return *this;
        }

        /**
         * @brief Set the stream output for the logger
         * 
         * @param opt_stream output stream
         * @return Logger& logger instance
         */
        Logger &setStream(std::ostream *opt_stream)
        {
            os = opt_stream;
            return *this;
        }

        /**
         * @brief Set the logger level
         * 
         * @param l level type enumerator
         * @return Logger& logger instance
         */
        Logger &setLevel(log_level l)
        {
            thresholdLevel = l;
            return *this;
        }

        /**
         * @brief Set the logger's messages prefix
         * 
         * @param s prefix string
         * @return Logger& logger instance
         */
        Logger &setPrefix(std::string s)
        {
            prefix = s;
            return *this;
        }

        template <typename T>
        Logger &operator<<(const T &x)
        {
            if ((int)thresholdLevel <= (int)currentType) {
                *os << x;
            }

            return *this;
        }

        Logger &operator<<(std::ostream &(*f)(std::ostream &o))
        {
            if ((int)thresholdLevel <= (int)currentType) {
                *os << f;
            }

            return *this;
        };

    private:
        /**
         * @brief Construct a new Logger, currently the output stream is forced to be
         * the standard output
         * 
         */
        Logger() : os(&std::cout)
        {
            resetImpl();
        }

        /**
         * @brief Reset function implementation
         */
        void resetImpl()
        {
            prefix = "";
            thresholdLevel = log_level::NORMAL;
        }

        Logger(const Logger &) = delete;
        Logger &operator=(const Logger &) = delete;

        std::ostream *os;
        std::string prefix;
        log_level thresholdLevel;
        log_type currentType;
    };

} // namespace mpc