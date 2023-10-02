/** @file sys_main.c 
 *   @brief Application main file
 *   @date 11-Dec-2018
 *   @version 04.07.01
 *
 *   This file contains an empty main function,
 *   which can be used for the application.
 */

/* 
 * Copyright (C) 2009-2018 Texas Instruments Incorporated - www.ti.com
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


/* USER CODE BEGIN (0) */
#include <adc.h>
#include <het.h>
#include <sci.h>
#include <math.h>
#include <stdio.h>
#include <FreeRTOS.h>
#include <os_task.h>
#include <os_queue.h>
#include <os_semphr.h>
/* USER CODE END */

/* Include Files */

#include "sys_common.h"

/* USER CODE BEGIN (1) */

//-------------------------------------------------------------------------------------------------
// Forward declarations of the RTOS task functions
//-------------------------------------------------------------------------------------------------

void adc_task(void* pvParameters);
void bldc_task(void *pvParameters);
void serial_task(void *pvParameters);
void encoder_task(void* pvParameters);
void identification_task(void *pvParameters);

inline int getAdcVoltage(adcData_t* data, const int channel) {
    int i;
    for (i = 0; i < 5; ++i) {
        if (data[i].id == channel)
            return data[i].value;
    }

    return 0;
}

inline int map(const int x,
               const int in_min, const int in_max,
               const int out_min, const int out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

//-------------------------------------------------------------------------------------------------
// Operational constants
//-------------------------------------------------------------------------------------------------

/**
 * @def DISTANCE
 * @brief Defines the distance between the disc rotor & the magnets (in mm)
 */
#define DISTANCE 10

/**
 * @def ENABLE_SYSTEM_IDENTIFICATION
 * @brief Enables or disables the system identification task
 */
#define ENABLE_SYSTEM_IDENTIFICATION 1

//-------------------------------------------------------------------------------------------------
// Utility pseudo-functions
//-------------------------------------------------------------------------------------------------

/**
 * @def max(a, b)
 * @brief Computes the maximum of two values.
 *
 * This macro compares two values and returns the greater of the two.
 *
 * @param a First value to be compared.
 * @param b Second value to be compared.
 * @return The maximum value between a and b.
 */
#define max(a, b) ((a) > (b) ? (a) : (b))

/**
 * @def min(a, b)
 * @brief Computes the minimum of two values.
 *
 * This macro compares two values and returns the lesser of the two.
 *
 * @param a First value to be compared.
 * @param b Second value to be compared.
 * @return The minimum value between a and b.
 */
#define min(a, b) ((a) < (b) ? (a) : (b))

//-------------------------------------------------------------------------------------------------
// System state structures
//-------------------------------------------------------------------------------------------------

/**
 * @enum test_mode
 * @brief Enumeration for the different test modes.
 */
typedef enum {
    POT_CONTROL   = 0, /**< Control the motor through potentiometer. */
    STEP_FUNCTION = 1, /**< Use a step function for system identification. */
    RAMP_FUNCTION = 2, /**< Use a ramp function for system identification. */
    SINUSOIDAL    = 3, /**< Use a sinusoidal function for system identification. */
    STOPPED       = 4, /**< Indicates that the motor was stopped, used for separating CSV data */
} TestMode;

/**
 * @struct state_t
 * @brief Struct to encapsulate the current state of the system
 */
typedef struct {
    int periodA;        /**< The RPM of the motor. */
    int periodB;        /**< The RPM of the motor. */
    int temperature;    /**< The temperature of the magnetic brake */
    int currentMtrA;    /**< The average current reading. */
    int currentMtrB;    /**< The average current reading. */
    int currentMtrC;    /**< The average current reading. */
    int totalCurrent;   /**< The average current reading. */
    int reference;      /**< Duty cycle of the ESC. */
    float distance;     /**< Distance setting in mm during the test. */
    TestMode mode;      /**< Current operation mode. */
} SystemState;

//-------------------------------------------------------------------------------------------------
// Global variables
//-------------------------------------------------------------------------------------------------

/**
 * @brief Represents the current state of the system.
 *
 * The SystemState structure holds real-time information regarding the system.
 * This includes but is not limited to, parameters like RPM, current, and reference values.
 */
static SystemState STATE;

/**
 * @brief A semaphore used to ensure mutual exclusion when accessing shared resources.
 *
 * MUTEX is particularly useful when multiple tasks or threads might access
 * or modify shared resources (like STATE) concurrently. By using this semaphore,
 * the system can avoid race conditions and ensure data consistency.
 */
static SemaphoreHandle_t MUTEX;


/* USER CODE END */

/** @fn void main(void)
 *   @brief Application main function
 *   @note This function is empty by default.
 *
 *   This function is called after startup.
 *   The user can use this function to implement the application.
 */

/* USER CODE BEGIN (2) */
/* USER CODE END */

int main(void)
{
    /* USER CODE BEGIN (3) */
    // Initialize modules
    adcInit();
    hetInit();
    sciInit();

    // Set initial state
    STATE.periodA = 0;
    STATE.periodB = 0;
    STATE.reference = 0;
    STATE.currentMtrA = 0;
    STATE.currentMtrB = 0;
    STATE.currentMtrC = 0;
    STATE.totalCurrent = 0;
    STATE.distance = DISTANCE;
    if (ENABLE_SYSTEM_IDENTIFICATION)
        STATE.mode = STEP_FUNCTION;
    else
        STATE.mode = POT_CONTROL;

    // Create mutex for sharing state between multiple tasks
    MUTEX = xSemaphoreCreateMutex();

    // Initialize common tasks
    xTaskCreate(adc_task,
                "ADC readings",
                1024,
                NULL,
                2,
                NULL);
    xTaskCreate(bldc_task,
                "BLDC motor control",
                256,
                NULL,
                2,
                NULL);
    xTaskCreate(encoder_task,
                "Encoder reading",
                1024,
                NULL,
                2,
                NULL);
    xTaskCreate(serial_task,
                "Transmit the state of the system",
                512,
                NULL,
                2,
                NULL);

    // Initialize system identification task
    if (ENABLE_SYSTEM_IDENTIFICATION) {
        xTaskCreate(identification_task,
                    "System identification task",
                    512,
                    NULL,
                    1,
                    NULL);
    }

    // Start the scheduler
    vTaskStartScheduler();
    while (1);

    /* USER CODE END */

    return 0;
}


/* USER CODE BEGIN (4) */
/**
 * @brief Serial task to transmit the state of the system.
 *
 * This task reads the shared state of the system (shared between multiple tasks)
 * and sends this data out through the serial port in a CSV format.
 *
 * @param pvParameters Parameters for the task (unused).
 * @note The shared system state (represented by the `STATE` variable) is accessed
 *       within mutex protection to ensure data integrity.
 */
void serial_task(void *pvParameters) {
    // Initialize task variables
    uint8 buffer[1024];
    const char* format = "$%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d;\r\n";

    // Reference, Distance, Total Current, Current Motor A, Current Motor B, Current Motor C, Temperature, Period A, Period B, Mode, Runtime*/

    // Wait for sensor calibration to end
    vTaskDelay(pdMS_TO_TICKS(15000));

    // Periodic instructions
    while (true) {
        // Lock mutex for reading shared state
        if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
            // Get runtime in milliseconds
            TickType_t ticks = xTaskGetTickCount();
            uint32 runtime_ms = (ticks * 1000) / configTICK_RATE_HZ;

            // Update buffer
            uint16 len = snprintf((char*) buffer, sizeof(buffer), format,
                                  (int) STATE.reference,
                                  (int) STATE.distance,
                                  (int) STATE.totalCurrent,
                                  (int) STATE.currentMtrA,
                                  (int) STATE.currentMtrB,
                                  (int) STATE.currentMtrC,
                                  (int) STATE.temperature,
                                  (int) STATE.periodA,
                                  (int) STATE.periodB,
                                  (int) STATE.mode,
                                  (unsigned int) runtime_ms);

            // Unlock mutex
            xSemaphoreGive(MUTEX);

            // Send buffer
            sciSend(scilinREG, len, buffer);
        }

        // Wait a little
        vTaskDelay(pdMS_TO_TICKS(20));
    }

}

/**
 * @brief Task for ADC readings.
 * @param pvParameters Task parameters (not used).
 */
void adc_task(void *pvParameters) {
    // Initialize HAL structures
    adcData_t adc[5];

    // Define periodic task instructions
    while (1) {
        // Start & wait for ADC conversion
        adcStartConversion(adcREG1, adcGROUP1);
        while (!adcIsConversionComplete(adcREG1, adcGROUP1));

        // Fetch the conversion results
        adcGetData(adcREG1, adcGROUP1, adc);

        // Get ADC readings
        float currentMtrA = fabs(getAdcVoltage(adc, 0));
        float currentMtrB = fabs(getAdcVoltage(adc, 1));
        float currentMtrC = fabs(getAdcVoltage(adc, 2));
        float totlCurrent = fabs(getAdcVoltage(adc, 3));
        float temperature = fabs(getAdcVoltage(adc, 4));

        // Update shared structure
        if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
            STATE.currentMtrA = (int) currentMtrA;
            STATE.currentMtrB = (int) currentMtrB;
            STATE.currentMtrC = (int) currentMtrC;
            STATE.temperature = (int) temperature;
            STATE.totalCurrent = (int) totlCurrent;
            xSemaphoreGive(MUTEX);
        }

        // Wait a little
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

/**
 * @brief Task for controlling the BLDC motor.
 *
 * This task is responsible for:
 * - Calculating the PWM duty cycle based on the desired reference
 * - Configuring the HET RAM for the PWM generation for the BLDC motor's ESC
 * - Updating the global state structure with the calculated duty cycle
 *
 * @param pvParameters Task parameters (not used).
 */
void bldc_task(void *pvParameters) {
    // Initialize HET parameters for communicating with the ESC
    uint32 duty = 0;
    uint32 action = 0;
    float64 pwmPeriod = 0;
    uint32 pwmPolarity = 0;

    // Define periodic task instructions
    while (1) {
        if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
            // Calculate the duty cycle for the ESC controller
            duty = map(STATE.reference, 0.0f, 100.0f, 500.0f, 750.0f);
            xSemaphoreGive(MUTEX);

            // Calculate PWM period and polarity specifics for HET RAM configurations
            pwmPolarity = 3U;
            pwmPeriod = (20 * 1000.0f * 1000.0f) / 640.000f;
            if (duty == 0U)
                action = 0U;
            else
                action = pwmPolarity;

            // Set the PWM configuration in the HET RAM
            hetRAM1->Instruction[(pwm0 << 1U) + 41U].Control = ((hetRAM1->Instruction[(pwm0 << 1U) + 41U].Control) & (~(uint32)(0x00000018U))) | (action << 3U);
            hetRAM1->Instruction[(pwm0 << 1U) + 41U].Data = ((((uint32)pwmPeriod * duty) / 10000U) << 7U) + 128U;
            hetRAM1->Instruction[(pwm0 << 1U) + 42U].Data = ((uint32)pwmPeriod << 7U) - 128U;
        }

        // Wait a little
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

/**
 * @brief Task for reading and filtering the encoder RPM values.
 *
 * This task is responsible for:
 * - Reading the RPM from the encoder using the HET capture signal
 * - Updating the global state structure with the calculated RPM value
 *
 * @param pvParameters Task parameters (not used).
 */
void encoder_task(void *pvParameters) {
    // Initialize HAL parameters
    hetSIGNAL_t encoderSignalA;
    hetSIGNAL_t encoderSignalB;

    // Define periodic task instructions
    while (1) {
        // Read both encoders
        capGetSignal(hetRAM1, cap0, &encoderSignalA);
        capGetSignal(hetRAM1, cap1, &encoderSignalB);

        // Get period from encoders
        float periodA = encoderSignalA.period;
        float periodB = encoderSignalB.period;

        // Update shared structure
        if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
            STATE.periodA = (int) periodA;
            STATE.periodB = (int) periodB;
            xSemaphoreGive(MUTEX);
        }

        // Wait a little
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

/**
 * @brief Performs system identification on the motor.
 *
 * This task is designed to identify the response of a motor to various input signals:
 * a step function, a ramp, and a sinusoidal wave. After each identification sequence,
 * the motor is stopped for a short duration.
 *
 * The system identification process involves:
 * 1. Stopping the motor.
 * 2. Applying a step function input.
 * 3. Stopping the motor again.
 * 4. Applying a ramp function input.
 * 5. Stopping the motor.
 * 6. Applying a sinusoidal input for a predefined number of repetitions.
 * 7. Finally, stopping the motor.
 *
 * @param pvParameters Standard task parameter (not used).
 *
 * @note The shared system state (represented by the `STATE` variable) is accessed
 *       within mutex protection to ensure data integrity.
 */
void identification_task(void *pvParameters) {
    // Define the number of times each test is performed
    int i, j, k;
    const int totalOscillations = 5;

    // Stop-start-stop motor to calibrate encoder periods
    if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
        STATE.reference = 0;
        STATE.mode = STOPPED;
        xSemaphoreGive(MUTEX);
        vTaskDelay(pdMS_TO_TICKS(2000));

        STATE.reference = 100;
        STATE.mode = STEP_FUNCTION;
        xSemaphoreGive(MUTEX);
        vTaskDelay(pdMS_TO_TICKS(1000));

        STATE.reference = 0;
        STATE.mode = STOPPED;
        xSemaphoreGive(MUTEX);
        vTaskDelay(pdMS_TO_TICKS(15000));
    }

    // Step function @ from 20% to 100%
    for (i = 1; i < 10; ++i) {
        for (j = 0; j < totalOscillations; ++j) {
            if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
                STATE.reference = (i + 1) * 10;
                STATE.mode = STEP_FUNCTION;
                xSemaphoreGive(MUTEX);
                vTaskDelay(pdMS_TO_TICKS(10000));
            }

            if (xSemaphoreTake(MUTEX, portMAX_DELAY)) {
                STATE.reference = 0;
                STATE.mode = STOPPED;
                xSemaphoreGive(MUTEX);
                vTaskDelay(pdMS_TO_TICKS(10000));
            }
        }
    }

    // Suspend all tasks so that Serial Studio automatically
    // closes the CSV file
    vTaskSuspendAll();
}
/* USER CODE END */
